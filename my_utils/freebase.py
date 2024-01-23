import json
from email.mime import base
from unittest import result
from xml.sax.saxutils import quoteattr
import pyodbc
import time
from tqdm import tqdm
from typing import List, Dict, Set, Tuple

from my_utils.logger import Logger
from config import Config

# 设置全局变量控制输出一些信息
PRINT_QUERY = False
SAMPLE_NUM = 3

forward_query = lambda base_query, connect_var, value_constraint: f'''SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?p
WHERE {{
{base_query}
{connect_var} ?p ?o.
{value_constraint}
FILTER (strstarts(str(?p), "http://rdf.freebase.com/ns/"))
}}
'''

reverse_query = lambda base_query, connect_var, value_constraint: f'''SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?pr
WHERE {{
{base_query}
?o ?pr {connect_var}.
{value_constraint}
FILTER (strstarts(str(?pr), "http://rdf.freebase.com/ns/"))
}}
'''

conn_query = lambda p, trip, candi : f'''SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?p
WHERE {{
    ?s {p} ?o.
    {trip}
    VALUES ?p {{{candi}}}
}}
'''

class FreebaseODBC():
    logger = Logger.get_logger(name="FreebaseODBC")

    def __init__(self, timeout=Config.query_time_limit):
        self.endpoint = pyodbc.connect('DSN=freebase;UID=dba;PWD=dba')
        self.endpoint.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        self.endpoint.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        self.endpoint.setencoding(encoding='utf-8')
        self.endpoint.timeout = timeout
        self.cursor = self.endpoint.cursor()

    @classmethod
    def __shorten(cls, uri:str)->str:
        if uri is None:
            return None
        return uri.replace('http://rdf.freebase.com/ns/', 'ns:')
    
    @classmethod
    def __is_ns_prefix(cls, prop:str)->bool:
        return prop.startswith("http://rdf.freebase.com/ns/")

    def query_answers(self, sparql:str):
        with self.endpoint:
            self.cursor.execute(sparql)
        rows = self.cursor.fetchall()
        ans = []
        for row in rows:
            ans.append(self.__shorten(row[0]))
        return ans
    
    # 为大查询准备的单独的 endpoint
    @classmethod 
    def get_new_endpoint(cls, timeout=10000):
        fail = True
        while fail:
            try:
                endpoint = pyodbc.connect('DSN=freebase;UID=dba;PWD=dba')
                endpoint.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                endpoint.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                endpoint.setencoding(encoding='utf-8')
                endpoint.timeout = timeout
                fail = False
            except:
                time.sleep(10)
                cls.logger.warn("[get_new_endpoint] failed to get new endpoint.")
        return endpoint

    @classmethod
    def query_pred_info(cls, timeout=1000):
        print("[INFO] query freebase predicate info...")
        endpoint = FreebaseODBC.get_new_endpoint()
        cursor = endpoint.cursor()

        propQuery = """
                SPARQL
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT DISTINCT ?p where {
                ?s ?p ?o.
                }
                """
        with endpoint:
            cursor.execute(propQuery)
        rows = cursor.fetchall()
        props = []
        for row in rows:
            if FreebaseODBC.__is_ns_prefix(row.p):
                props.append(cls.__shorten(row.p))
        
        infoQ = lambda x: f"""
                SPARQL
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT ?p ?domain ?range ?name ?revP where {{
                optional{{?p rdfs:domain ?domain.}}
                optional{{?p rdfs:range ?range.}}
                optional{{?p rdfs:label ?name.}}
                optional{{?p owl:inverseOf ?revP.}}
                FILTER (?p={x})
                }}
                """
        prop_info_dict = dict()
        for prop in tqdm(props):
            with endpoint:
                cursor.execute(infoQ(prop))
            row = cursor.fetchone()
            # 丢掉啥信息都没有的谓词
            if row == None:
                continue
            prop_info_dict[prop]={"domain":cls.__shorten(row.domain),"range":cls.__shorten(row.range),"name":row.name,"reverse":cls.__shorten(row.revP)}
        cursor.close()
        endpoint.close()
        return prop_info_dict
    
    @classmethod
    def query_type_info(self, type_list):
        print("[INFO] query freebase type info...")
        endpoint = FreebaseODBC.get_new_endpoint()
        cursor = endpoint.cursor()

        infoQ = lambda tp : f'''SPARQL
        select ?label ?i_count ?i_name_count
        where {{
            {{select count(?i) as ?i_count where {{ ?i a {tp}.}} }}
            {{select count(?i) as ?i_name_count {{ select distinct ?i where {{?i a {tp}. ?i ns:type.object.name [].}} }} }}
            optional {{ {tp} rdfs:label ?label. filter(lang(?label)='en')}}
        }}
        '''
        type_info_dict = dict()
        for tp in tqdm(type_list):
            with endpoint:
                cursor.execute(infoQ(tp))
            row = cursor.fetchone()
            type_info_dict[tp] = {"name":row.label, "instance_count":row.i_count, "has_name_instance_count":row.i_name_count}
            assert row.i_count >= row.i_name_count
        cursor.close()
        endpoint.close()
        return type_info_dict
    
    @classmethod
    def query_neighbor_preds(cls, base_path:List[str])->List[str]:
        results = []
        base_query, connect_var = cls.__build_base_path_query(base_path)
        query1 = forward_query(base_query, connect_var, "")
        query2 = reverse_query(base_query, connect_var, "")
        # print(query1)
        # print(query2)
        conn = cls.get_new_endpoint(Config.query_time_limit)
        cursor = conn.cursor()
        try:
            with conn:
                cursor.execute(query1)
            for row in cursor.fetchall():
                results.append(cls.__shorten(row.p))
        except Exception as e:
            cls.logger.debug(f"[forward neighbor] base_query: {base_query}, error_info: {e}")
            
        try:
            with conn:
                cursor.execute(query2)
            for row in cursor.fetchall():
                results.append(cls.__shorten(row.pr)+"_Rev")
        except Exception as e:
            cls.logger.debug(f"[reverse neighbor] base_query: {base_query}, error_info: {e}")
        # print(results)
        cursor.close()
        conn.close()
        return results
    
    def query_predstr_to_ans(self, base_path:List[str], nary_preds:List[str], ans_ents:Set[str])->List[str]:
        results = []
        ans_ent = list(ans_ents)[0] # !!! 这里相当于随便取了一个答案节点用于搜索
        # reach answer by binary pred
        base_query, connect_var = self.__build_base_path_query(base_path)
        query1 = forward_query(base_query, connect_var, f"VALUES ?o {{{ans_ent}}}")
        query2 = reverse_query(base_query, connect_var, f"VALUES ?o {{{ans_ent}}}")
        try:
            with self.endpoint:
                self.cursor.execute(query1)
            for row in self.cursor.fetchall():
                results.append(self.__shorten(row.p))
        except Exception as e:
            self.logger.debug(f"[forward answer] base_query: {base_query}, ans_ent: {ans_ent}, error_info: {e}")
        
        try:
            with self.endpoint:
                self.cursor.execute(query2)
            for row in self.cursor.fetchall():
                results.append(self.__shorten(row.pr)+"_Rev")
        except Exception as e:
            self.logger.debug(f"[reverse answer] base_query: {base_query}, ans_ent: {ans_ent}, error_info: {e}")
            
        # reach answer by n-ary pred
        for pred in nary_preds:
            base_query, connect_var = self.__build_base_path_query(base_path+[pred])
            query1 = forward_query(base_query, connect_var, f"VALUES ?o {{{ans_ent}}}")
            query2 = reverse_query(base_query, connect_var, f"VALUES ?o {{{ans_ent}}}")
            try:
                with self.endpoint:
                    self.cursor.execute(query1)
                for row in self.cursor.fetchall():
                    results.append(pred + "..." + self.__shorten(row.p))
            except Exception as e:
                self.logger.debug(f"[forward answer] base_query: {base_query}, ans_ent: {ans_ent}, error_info: {e}")
                
            try:
                with self.endpoint:
                    self.cursor.execute(query2)
                for row in self.cursor.fetchall():
                    results.append(pred + "..." + self.__shorten(row.pr)+"_Rev")
            except Exception as e:
                self.logger.debug(f"[reverse answer] base_query: {base_query}, ans_ent: {ans_ent}, error_info: {e}")
                
        return results
    
    def query_nary_predstrs(self, base_path:List[str], nary_preds:List[str])->List[str]:
        results = []
        for pred in nary_preds:
            base_query, connect_var = self.__build_base_path_query(base_path+[pred])
            query1 = forward_query(base_query, connect_var, "")
            query2 = reverse_query(base_query, connect_var, "")
            try:
                with self.endpoint:
                    self.cursor.execute(query1)
                for row in self.cursor.fetchall():
                    results.append(pred + "..." + self.__shorten(row.p))
            except Exception as e:
                self.logger.debug(f"[forward n-ary] base_query: {base_query}, error_info: {e}")
                
            try:
                with self.endpoint:
                    self.cursor.execute(query2)
                for row in self.cursor.fetchall():
                    results.append(pred + "..." + self.__shorten(row.pr)+"_Rev")
            except Exception as e:
                self.logger.debug(f"[reverse n-ary] base_query: {base_query}, error_info: {e}")
                
        return results  
    
    @classmethod
    def query_binary_facts(self, base_path:List[str], pred:str) -> List[List[str]]:
        ans = []
        base_query, connect_var = self.__build_base_path_query(base_path)
        if base_query == "":
            base_query = f"VALUES ?x {{{connect_var}}}"
        p = pred
        if pred.endswith("_Rev"):
            p = "^"+pred[:-4]
        query = f""" SPARQL
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?x ?o
        WHERE {{
        {base_query}
        ?x {p} ?o.
        }}
        """
        # print(query)
        conn = self.get_new_endpoint(Config.query_time_limit)
        cursor = conn.cursor()
        try:
            with conn:
                cursor.execute(query)
            for row in cursor.fetchall():
                ans.append([self.__shorten(row.x), pred, self.__shorten(row.o)])
        except Exception as e:
            self.logger.debug(f"[binary facts] base_query: {base_query}, pred: {pred}, error_info: {e}")
        cursor.close()
        conn.close()
        return ans
    
    @classmethod
    def query_nary_facts(self, base_path:List[str], pred1:str, pred2:str) -> List[List[str]]:
        ans = []
        base_query, connect_var = self.__build_base_path_query(base_path)
        if base_query == "":
            base_query = f"VALUES ?x {{{connect_var}}}"
        p1 = pred1
        p2 = pred2
        if pred1.endswith("_Rev"):
            p1 = "^"+pred1[:-4]
        if pred2.endswith("_Rev"):
            p2 = "^"+pred2[:-4]
        query = f""" SPARQL
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?x ?o1 ?o
        WHERE {{
        {base_query}
        ?x {p1} ?o1.
        ?o1 {p2} ?o.
        }}
        """
        conn = self.get_new_endpoint(Config.query_time_limit)
        cursor = conn.cursor()
        try:
            with conn:
                cursor.execute(query)
            for row in cursor.fetchall():
                ans.append([self.__shorten(row.x), pred1, self.__shorten(row.o1), pred2, self.__shorten(row.o)])
        except Exception as e:
            self.logger.debug(f"[n-ary facts] base_query: {base_query}, pred1: {pred1}, pred2: {pred2}, error_info: {e}")
        cursor.close()
        conn.close()
        return ans
    
    @classmethod
    def __build_base_path_query(self, base_path:List[str])->Tuple:
        assert len(base_path) > 0
        if len(base_path) == 1:
            return "", base_path[0]
        else:
            start = base_path[0]
            preds = []
            for item in base_path[1::]:
                for pred in item.split("..."):
                    if pred.endswith("_Rev"):
                        preds.append("^"+pred[:-4])
                    else:
                        preds.append(pred)
            return f"{start} {'/'.join(preds)} ?x.", "?x"
        
    @classmethod
    def query_pred_conn(cls, pred:str):
        # 从全局的角度查询关系之间的连接关系
        pred_conn_info = lambda conn_trip, pred , var : f'''SPARQL
        select distinct ?p
        where {{
        {{
            select distinct {var}
            where {{
            ?s {pred} ?o.
            Filter (strstarts(str({var}),"http://rdf.freebase.com/ns/"))
            }}
            limit 10000000
        }}
        {conn_trip}
        Filter (strstarts(str(?p),"http://rdf.freebase.com/ns/"))
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
    
    # @classmethod
    # def query_pred_conn(cls, pred:str, candi_pred:List[str]) -> Dict[str, List[str]]:
    #     ss_query = conn_query(pred, "?s ?p ?x.", " ".join(candi_pred))
    #     so_query = conn_query(pred, "?x ?p ?s.", " ".join(candi_pred))
    #     os_query = conn_query(pred, "?o ?p ?x.", " ".join(candi_pred))
    #     oo_query = conn_query(pred, "?x ?p ?o.", " ".join(candi_pred))
    #     ans = {"S-S":[], "S-O":[], "O-S":[], "O-O":[]}
    #     # print(ss_query)
    #     # print(so_query)
        
    #     conn = cls.get_new_endpoint(60)
    #     cursor = conn.cursor()
        
    #     with conn:
    #         cursor.execute(ss_query)
    #     for row in cursor.fetchall():
    #         ans["S-S"].append(row.p)
            
    #     with conn:
    #         cursor.execute(so_query)
    #     for row in cursor.fetchall():
    #         ans["S-O"].append(row.p)
            
    #     with conn:
    #         cursor.execute(os_query)
    #     for row in cursor.fetchall():
    #         ans["O-S"].append(row.p)
            
    #     with conn:
    #         cursor.execute(oo_query)
    #     for row in cursor.fetchall():
    #         ans["O-O"].append(row.p)
        
    #     cursor.close()
    #     conn.close()
        
    #     return ans
    
    @classmethod
    def __query_and_filter_pred(cls, query:str):
        # print(query)
        conn = cls.get_new_endpoint(timeout=1)
        cursor = conn.cursor()
        with conn:
            cursor.execute(query)
        rows = cursor.fetchall()
        ans = []
        for row in rows:
            if cls.__is_ns_prefix(row.p):
                ans.append(cls.__shorten(row.p))
        cursor.close()
        conn.close()
        return ans
    
    @classmethod
    def has_query_results(cls, query:str) -> bool:
        conn = cls.get_new_endpoint(1)
        cursor = conn.cursor()
        rows = []
        try:
            with conn:
                cursor.execute(query)
            rows = cursor.fetchall()
        except Exception as e:
            cls.logger.debug(f"[has_query_results] query:{query}, err_info:{e}")
        cursor.close()
        conn.close()
        return len(rows) != 0
    
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
            cls.logger.debug(f"[get_query_results] query:{query}, err_info:{e}")
        cursor.close()
        conn.close()
        for row in rows:
            for idx, var in enumerate(vars):
                ans[var].append(cls.__shorten(row[idx]))
        return ans


def query_pred_conn_for_freebase():
    myodbc = FreebaseODBC()
    with open(Config.cache_pred_info, 'r') as f:
        pred_info_fb = json.load(f)
    pred_list = list(pred_info_fb.keys())
    for pred in tqdm(pred_list):
        info = myodbc.query_pred_conn(pred)
        res = {"id": pred, "pred_conn": info}
        with open('/home/jjyu/IRQA/data/cache/freebase_props_conn_cache.jsonl', 'a') as wf:
            wf.write(json.dumps(res) + '\n')
    print(1)


if __name__ == "__main__":
    # my_odbc = FreebaseODBC()
    # query = "SPARQL PREFIX ns: <http://rdf.freebase.com/ns/> select ?j where {ns:m.02vymvp ns:business.board_member.leader_of ?j.}"
    # result = my_odbc.get_query_results(query, 'j')
    pass
