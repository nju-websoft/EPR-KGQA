import json

import re

import pyodbc
import time
from typing import List, Dict, Set, Tuple
from tqdm import tqdm

tqdm.monitor_interval = 0

class WikidataODBC():
    def __init__(self, timeout=1000):
        self.endpoint = pyodbc.connect('DRIVER=/usr/local/lib/virtodbc.so;Host=114.212.81.217:1115;UID=dba;PWD=dba')
        self.endpoint.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        self.endpoint.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        self.endpoint.setencoding(encoding='utf-8')
        self.endpoint.timeout = timeout
        self.cursor = self.endpoint.cursor

    @classmethod
    def get_new_endpoint(cls, timeout=10000):
        fail = True
        while fail:
            try:
                endpoint = pyodbc.connect('DRIVER=/usr/local/lib/virtodbc.so;Host=114.212.81.217:1115;UID=dba;PWD=dba')
                endpoint.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                endpoint.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                endpoint.setencoding(encoding='utf-8')
                endpoint.timeout = timeout
                fail = False
            except Exception as e:
                time.sleep(10)
                cls.logger.warn("[get_new_endpoint] failed to get new endpoint.")
        return endpoint

    @classmethod
    def get_query_results(cls, query, vars) -> Dict[str, List[str]]:
        ans = dict()
        for var in vars:
            ans[var] = []
        conn = cls.get_new_endpoint(1)
        cursor = conn.cursor()
        rows = []
        try:
            with conn:
                cursor.execute(query)
            rows = cursor.fetchall()
        except Exception as e:
            print(e)
        cursor.close()
        conn.close()
        for row in rows:
            for idx, var in enumerate(vars):
                ans[var].append(row[idx])
        return ans

    @classmethod
    def query_all_preds(cls, timeout=2000):
        endpoint = cls.get_new_endpoint(timeout)
        cursor = endpoint.cursor()
        query = """
                SPARQL
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                SELECT DISTINCT ?p where {
                ?s ?p ?o.
                }
                """
        with endpoint:
            cursor.execute(query)
        count = 0
        rows = []
        while True:
            try:
                row = cursor.fetchone()
            except Exception as e:
                print(e)
                continue
            if not row:
                break
            prop = row[0]
            count += 1
            if count % 1000 == 0:
                print(count)
            # with open(os.path.join(data_path, 'wikidata_preds.txt'), 'a+') as f:
            #     f.write(str(list(row)) + '\r\n')
            rows.append(prop)

        # rows = cursor.fetchall()
        cursor.close()
        return rows

    @classmethod
    def __shorten(cls, uri: str) -> str:
        if uri is None:
            return None
        return uri.replace('http://www.wikidata.org/prop/direct/', '')

    @classmethod
    def __is_wiki_prop_prefix(cls, prop: str) -> bool:
        return prop.startswith("http://www.wikidata.org/prop/direct/")

    @classmethod
    def __query_and_filter_pred(cls, query: str):
        conn = cls.get_new_endpoint(timeout=10000)
        cursor = conn.cursor()
        try:
            with conn:
                cursor.execute(query)
        except Exception as e:
            directs = re.findall(r'wdt:(P[0-9]+)', query)
            assert len(directs) == 1
            with open('/home/jjyu/IRQA/data/cache/wikidata/error_props.jsonl', 'a') as wf:
                wf.write(directs[0] + '\n')
        rows = []
        while True:
            try:
                row = cursor.fetchone()
            except Exception as e:
                print(e)
                continue
            if not row:
                break
            prop = row[0]
            # with open(os.path.join(data_path, 'wikidata_preds.txt'), 'a+') as f:
            #     f.write(str(list(row)) + '\r\n')
            rows.append(prop[36:])
        cursor.close()
        conn.close()
        return rows

    @classmethod
    def query_pred_conn(cls, pred: str):
        # 从全局的角度查询关系之间的连接关系
        pred_conn_info = lambda conn_trip, pred, var: f'''SPARQL
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            select distinct ?p
            where {{
            {{
                select distinct {var}
                where {{
                ?s wdt:{pred} ?o.
                Filter (strstarts(str({var}), "http://www.wikidata.org/"))
                }}
                limit 10000000
            }}
            {conn_trip}
            Filter (strstarts(str(?p),"http://www.wikidata.org/prop/direct/"))
            }}
            '''
        ss_trip = '?s ?p ?t.'
        so_trip = '?t ?p ?s.'
        os_trip = '?o ?p ?t.'
        oo_trip = '?t ?p ?o.'
        ans = dict()
        ans["S-S"] = cls.__query_and_filter_pred(pred_conn_info(ss_trip, pred, "?s"))
        ans["S-O"] = cls.__query_and_filter_pred(pred_conn_info(so_trip, pred, "?s"))
        ans["O-S"] = cls.__query_and_filter_pred(pred_conn_info(os_trip, pred, "?o"))
        ans["O-O"] = cls.__query_and_filter_pred(pred_conn_info(oo_trip, pred, "?o"))
        return ans


def query_pred_conn_for_wikidata():
    myodbc = WikidataODBC()
    with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_direct.json', 'r') as f:
        wiki_props = json.load(f)
    for prop in tqdm(wiki_props):
        info = myodbc.query_pred_conn(prop)
        res = {"id": prop, "pred_conn": info}
        with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_conn_cache.jsonl', 'a') as wf:
            wf.write(json.dumps(res) + '\n')
    print(1)


if __name__ == "__main__":
    # query_pred_conn_for_wikidata()
    wikidata = WikidataODBC()
    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    select distinct ?var3 {
    values ?var3{
    wd:Q615
    wd:Q161089
    }
    ?var3 wdt:P27 ?var1.	#梅西的国籍是？
    ?var1 wdt:P2250 ?var2}
    ORDER BY ASC(xsd:integer(?var2)) """
    wikidata.get_query_results(query, 'var3')
