import pandas as pd

path = "/Users/snehalr/Documents/Snehal/Swadhyaya/PyScripts"
file_path = path + '/export.xml'

df = pd.read_xml(file_path)

df.to_csv(path + '/export.csv', index=False)
