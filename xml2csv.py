import pandas as pd

path = "/Users/snehalr/Documents/Snehal/Swadhyaya/PyScripts"
file_path = path + '/export1.xml'

df = pd.read_xml(file_path)

df.to_csv(path + '/export1.csv', index=False)