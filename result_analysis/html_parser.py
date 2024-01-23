from bs4 import BeautifulSoup
import csv
import json
from tqdm import tqdm
import requests

import pandas as pd
from my_utils.wikidata import WikidataODBC

tqdm.monitor_interval = 0


def html_table_to_csv(html_content, csv_filename):
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 查找表格
    table = soup.find('table')

    # 初始化CSV写入器
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 遍历表格行
        for row in table.find_all('tr'):
            # 获取行中的单元格
            cells = row.find_all(['th', 'td'])
            # 提取单元格文本并写入CSV文件
            csvwriter.writerow([cell.get_text(strip=True) for cell in cells])


def generate_wikidata_tables():
    # 替换为您的网页链接
    url = 'https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all'

    # 发送HTTP请求并获取页面内容
    response = requests.get(url)
    html_content = response.text

    # 指定CSV文件名
    csv_filename = "/home/jjyu/IRQA/data/cache/wikidata/wikidata_tables.csv"

    # 调用函数进行转换
    html_table_to_csv(html_content, csv_filename)

    print(f"HTML表格已成功转换为CSV文件：{csv_filename}")


def query_props_with_wi_datatype():
    df = pd.read_csv("/home/jjyu/IRQA/data/cache/wikidata/wikidata_tables.csv")
    df_wi = df.loc[df['Data type[1]']=='WI', ['ID']]
    df_wi_list = df_wi.values.tolist()
    props_wi = [item[0] for item in df_wi_list]
    myodbc = WikidataODBC()
    for prop in tqdm(props_wi):
        info = myodbc.query_pred_conn(prop)
        res = {"id": prop, "pred_conn": info}
        with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_wi_conn_cache.jsonl', 'a') as wf:
            wf.write(json.dumps(res) + '\n')
    print(1)


if __name__ == "__main__":
    query_props_with_wi_datatype()
    pass
