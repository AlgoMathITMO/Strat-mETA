import pandas as pd
from tensorflow import keras
from pandarallel import pandarallel
import numpy as np
import os
from datetime import datetime
import multiprocessing as mp
from tqdm.auto import tqdm
tqdm.pandas()
import requests
import psycopg2 as pg
from sklearn.model_selection import train_test_split

mp.set_start_method('fork', force=True)

from atpbar import register_reporter, find_reporter, flush
from atpbar import atpbar

class ValhallaAdapter:
    def __init__(self):
        self.valhalla_url = None

    def mapmatch(self, df):

        matched_osm_ids = []
        matched_dirs = []

        filter_params = {"attributes": ["edge.way_id", "matched.edge_index", "matched.type"],
                         "action": "include"}
        valhalla_template = dict()
        valhalla_template.update({"costing": "auto", "shape_match": "map_snap"})
        valhalla_template["filters"] = filter_params

        for track in atpbar(list(df.itertuples()), name = mp.current_process().name):
            shape = []

            for lat, lon, ts in zip(track.lats, track.lngs, track.ts):
                point_dict = {"lat": str(lat), "lon": str(lon), "time": str(ts)}
                shape.append(point_dict)
            valhalla_request = valhalla_template
            valhalla_request["shape"] = shape

            valhalla_response = dict()

            calls = 0
            while calls < 5:
                try:
                    valhalla_response_full = requests.post(self.valhalla_url, json=valhalla_request, timeout=5)
                except Exception as e:
                    calls += 1
                    print("MAPMATCHING " + f"Got error while requesting valhalla:\n{e}")
                else:
                    valhalla_response = valhalla_response_full.json()
                    break

            if calls == 5:
                matched_osm_ids.append(None)
                matched_dirs.append(None)
                continue

            try:
                if len(valhalla_response["edges"]) == 1:
                    matched_osm_ids.append(None)
                    matched_dirs.append(None)
                    continue
                else:
                    way_info = dict()
                    osm_ids = []
                    directions = []

                    for point_idx, matched_point in enumerate(valhalla_response["matched_points"]):

                        if matched_point["type"] == "unmatched":
                            osm_ids.append(None)
                            directions.append(-1)
                            continue

                        edge_index = matched_point["edge_index"]
                        if edge_index > 1000:
                            osm_ids.append(None)
                            directions.append(-1)
                            continue

                        edge_info = valhalla_response["edges"][edge_index]
                        osm_ids.append(edge_info['way_id'])
                        directions.append(int(edge_info['forward']))

                        # key = [edge_info['way_id'], edge_info['forward']]
                        # matched_points.append(key)

                    matched_osm_ids.append(osm_ids)
                    matched_dirs.append(directions)
            except Exception as e:
                continue

        return pd.DataFrame({'osm_ids': matched_osm_ids, 'durations': matched_dirs})

    def get_route_info(self, df):
        filter_params = {"attributes": ["edge.way_id", "matched.edge_index", "matched.type"],
                 "action": "include"}
        valhalla_template = dict()
        valhalla_template.update({"costing": "auto", "shape_match": "map_snap"})
        valhalla_template["filters"] = filter_params

        valhalla_durations = []
        distances = []

        for track in atpbar(list(df.itertuples()), name = mp.current_process().name):
            shape = []

            track_info = np.array(list(zip(track.lats, track.lngs, track.ts)))
            indexes = np.round(np.linspace(0, len(track_info) - 1, 10)).astype(int)
            for lat, lon, ts in track_info[indexes]:
                point_dict = {"lat": str(lat), "lon": str(lon), "time": str(ts), "type": "via"}
                shape.append(point_dict)
            valhalla_request = valhalla_template
            valhalla_request["locations"] = shape

            calls = 0
            while calls < 5:
                try:
                    valhalla_response_full = requests.post(self.valhalla_url, json=valhalla_request, timeout=5)
                except Exception as e:
                    calls += 1
                    print("MAPMATCHING " + f"Got error while requesting valhalla:\n{e}")
                else:
                    valhalla_response = valhalla_response_full.json()
                    break

            if calls == 5:
                valhalla_durations.append(None)
                distances.append(None)
                continue

            try:
                valhalla_duration = int(valhalla_response_full.json()['trip']['summary']['time'])
                distance = int(valhalla_response_full.json()['trip']['summary']['length'] * 1000)
                valhalla_durations.append(valhalla_duration)
                distances.append(distance)
            except Exception as e:
                valhalla_durations.append(None)
                distances.append(None)

        return pd.DataFrame({'valhalla_duration': valhalla_durations, 'distance': distances})

class Dataset:
    def __init__(self, csv_dir,
                       drop_columns=None,
                       rename_columns=None,
                       target_columns=None
                ):

        self.csv_dir_path = csv_dir
        self.filenames = os.listdir(csv_dir)
        self.filenames = list( filter(lambda x: x.split('.')[-1] == 'csv', self.filenames) )
        self.df = pd.DataFrame([], columns=target_columns)
        self.drop_columns = drop_columns
        self.rename_columns = rename_columns
        self.target_columns = target_columns

        self.valhalla_adapter = ValhallaAdapter()

        self.cong_idx_bounds = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.df[index]
        else:
            return self.df.iloc[[index]]

    def load(self, fname):
        return pd.read_csv( os.path.join(self.csv_dir_path, fname))
        
    def get_dataset(self, min_duration=600, min_points=10, drop_index=True):
        self.df = self.df.loc[
                np.logical_and(
                    self.df.n_points > min_points,
                    self.df.duration > min_duration
                             )][self.target_columns]

        return self.df.reset_index(drop=drop_index)

    def convert(self):

        for name in tqdm(self.filenames):
            dataset = pd.read_csv( os.path.join(self.csv_dir_path, name))
            if self.drop_columns:
                dataset = dataset.drop(columns=self.drop_columns)
            if self.rename_columns:
                dataset = dataset.rename(columns=self.rename_columns)

            dataset = dataset.progress_apply(self.__convert_df, axis=1)
            dataset = pd.concat(dataset.tolist(), axis=0, ignore_index=True)
            self.df = pd.concat([self.df, dataset])


    def __convert_df(self, row):

        def ts_split(ts):
            dt = 0
            bounds = [0]
            for idx in range(len(ts) - 1):
                dt = ts[idx] - ts[idx-1]
                if dt > 120:
                    bounds.append(idx)

            bounds.append(len(ts) - 1)
            return bounds

        ts = row.ts[1:-1]
        ts = ts.split(',')
        ts = list(map(int, ts))
        ts.reverse()

        bounds = ts_split(ts)
        tracks_ts = []
        tracks_lats = []
        tracks_lngs = []
        tracks_dur = []
        tracks_n_points = []
        tracks_start_date = []

        lats = row.lats[1:-1]
        lats = lats.split(',')
        lats = list(map(float, lats))
        lats.reverse()

        lngs = row.lngs[1:-1]
        lngs = lngs.split(',')
        lngs = list(map(float, lngs))
        lngs.reverse()

        for s_idx in range(len(bounds) - 1):
            start = bounds[s_idx]
            end = bounds[s_idx + 1]
            tracks_ts.append(ts[start:end])
            tracks_lats.append(lats[start:end])
            tracks_lngs.append(lngs[start:end])
            tracks_dur.append(ts[end] - ts[start])
            tracks_n_points.append(end - start)

            date = datetime.fromtimestamp(ts[start]).replace(second=0, microsecond=0)
            date = date.replace(minute=(date.minute // 10) * 10)
            tracks_start_date.append(date)

        return pd.DataFrame({'duration': tracks_dur, 'ts': tracks_ts, 'lats': tracks_lats, 'lngs': tracks_lngs, 'n_points': tracks_n_points, 'start_date': tracks_start_date})

    def to_feather(self, fname):
        self.df.to_feather(fname)

    def load_feather(self, fname,
                           columns=None,
                           use_threads=True):
        self.df = pd.read_feather(fname, columns=columns, use_threads=use_threads)
        self.drop_columns = None
        self.rename_columns = None
        self.target_columns = list(self.df.columns)

    def mapmatch(self, host='http://localhost:8002/trace_attribute', n_workers=16):

        self.valhalla_adapter.valhalla_url = host
        reporter = find_reporter()
        with mp.Pool(n_workers, register_reporter, [reporter]) as p:
            matched_tracks = p.map(self.valhalla_adapter.mapmatch, np.array_split(self.df, n_workers))
            matched_tracks = pd.concat(matched_tracks, axis=0, ignore_index=True)
            flush()

        self.df = pd.concat([self.df, matched_tracks], axis=1, ignore_index=True)
        self.df = self.df.rename(columns={0:'dur', 1:'ts', 2:'lats', 3:'lngs', 4:'n_points', 5:'start_ts', 6:'osm_ids', 7:'directions'})

    def get_route_info(self, host='http://localhost:8002/route', n_workers=16):
        self.valhalla_adapter.valhalla_url = host
        reporter = find_reporter()
        with mp.Pool(n_workers, register_reporter, [reporter]) as p:
            route_info = p.map(self.valhalla_adapter.get_route_info, np.array_split(self.df, n_workers))
            route_info = pd.concat(route_info, axis=0, ignore_index=True)
            flush()

        self.df = pd.concat([self.df, route_info], axis=1, ignore_index=True)
        self.df = self.df.rename(columns={0:'dur', 1:'ts', 2:'lats', 3:'lngs', 4:'n_points', 5:'start_ts', 6:'osm_ids', 7:'directions', 8:'cong_idx', 9:'valhalla_duration', 10:'distance'})

    def head(self, n=5):
        return self.df.head(n)

    def __classify_cong_indexes(self, df):
        cong_idx = df.cong_idx
        counts = []

        for b in self.cong_idx_bounds:
            mask = cong_idx <= b
            counts.append( len(cong_idx[mask]) )
            mask = cong_idx > b
            cong_idx = cong_idx[mask]
        counts.append(len(cong_idx))

        n_slow, n_cong, n_norm, n_free = counts
        df.n_slow = n_slow
        df.n_cong = n_cong
        df.n_norm = n_norm
        df.n_free = n_free

        return df

    def classify_cong_indexes(self, bounds):

        self.cong_idx_bounds = bounds

        self.df['n_slow'] = [0]*len(self.df)
        self.df['n_cong'] = [0]*len(self.df)
        self.df['n_norm'] = [0]*len(self.df)
        self.df['n_free'] = [0]*len(self.df)

        self.df = self.df.progress_apply(self.__classify_cong_indexes, axis=1)

    def __start_date_to_ts(self, df):
        df['start_hour'] = df.start_ts.hour
        return df

    def start_date_to_ts(self):
        self.df['start_hour'] = [0]*len(self.df)
        self.df = self.df.progress_apply(self.__start_date_to_ts, axis=1)

    def split(self, test_size, x_labels, y_labels, n=None, shuffle=True, rand_state=42):

        if shuffle:
            if n:
                df = self.df.sample(n=n, random_state=rand_state)
            else:
                df = self.df.sample(frac=1, random_state=rand_state)
        else:
            df = self.df[:n]

        X = df[x_labels]
        Y = df[y_labels]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=rand_state)

        return X_train, X_test, y_train, y_test

    def get_traffic_data(self, db_auth):

        self.df = self.df.sort_values(by=['start_ts'])

        traffic_scores_query = """with traffic_idx as (
                select ti.osm_id, ti.same_direction, ti.ri_rt_avg from traffic_scores, jsonb_to_recordset(traffic_scores.traffic_indexes)
                as ti(osm_id bigint, ri_rt_avg numeric, same_direction boolean)
                where ts='%s'::timestamp )
                select traffic_idx.* from traffic_idx left join traffic_indexes td
                on traffic_idx.osm_id = td.osm_id and traffic_idx.same_direction = td.same_direction;"""

        conn = pg.connect(**db_auth)

        cong_idx_col = []
        for date in tqdm(dataset.df.start_ts.unique()):
            cur = conn.cursor()
            cur.execute( traffic_scores_query % date )
            raw_traffic_data = cur.fetchall()
            cur.close()

            traffic_dict = dict()
            for osm_id, direction, cong_idx in raw_traffic_data:
                traffic_dict[(osm_id, direction)] = cong_idx

            for track in self.df.loc[self.df.start_ts == date].itertuples():
                track_cong_idx = []
                if not isinstance(track.osm_ids, np.ndarray):
                    cong_idx_col.append(None)
                    continue
                for osm_id, direction in zip(track.osm_ids, track.directions):
                    c_idx = traffic_dict.get((-osm_id, direction))
                    if c_idx:
                        track_cong_idx.append( int(c_idx*100) )
                    else:
                        track_cong_idx.append( 0 )

                cong_idx_col.append(track_cong_idx)

        conn.commit()
        conn.close()
        self.df = pd.concat([self.df, pd.DataFrame({'cong_idx': cong_idx_col})], axis=1, ignore_index=True)
        self.df = self.df.rename(columns={0:'dur', 1:'ts', 2:'lats', 3:'lngs', 4:'n_points', 5:'start_ts', 6:'osm_ids', 7:'directions', 8:'cong_idx'})
