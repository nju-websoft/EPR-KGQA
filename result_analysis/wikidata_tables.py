from bs4 import BeautifulSoup
import csv
import json
from tqdm import tqdm
import requests

import pandas as pd
from my_utils.wikidata import WikidataODBC

tqdm.monitor_interval = 0
df = pd.read_csv("/home/jjyu/IRQA/data/cache/wikidata/wikidata_tables.csv")
df_wi = df.loc[df['Data type[1]'] == 'WI', ['ID']]
df_wi_list = df_wi.values.tolist()
props_wi = [item[0] for item in df_wi_list]
props_wi.reverse()
myodbc = WikidataODBC()
for prop in tqdm(props_wi):
    info = myodbc.query_pred_conn(prop)
    res = {"id": prop, "pred_conn": info}
    with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_wi_conn_cache_reverse.jsonl', 'a') as wf:
        wf.write(json.dumps(res) + '\n')
print(1)


# 替换为您的网页链接
url = 'https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all'

# 发送HTTP请求并获取页面内容
response = requests.get(url)
html_content = response.text

# 使用Beautiful Soup解析HTML内容
soup = BeautifulSoup(html_content, 'html.parser')

# 找到页面中的所有表格
tables = soup.find_all('table')

# 遍历每个表格并提取数据
for table in tables:
    # 找到表格中的所有行
    rows = table.find_all('tr')

    for row in rows:
        # 找到行中的所有单元格
        cells = row.find_all(['th', 'td'])

        # 提取单元格中的文本内容并打印
        for cell in cells:
            print(cell.get_text(), end='\t')
        print()  # 换行表示新的一行