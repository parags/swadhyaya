import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse("/Users/snehalr/Documents/Snehal/Swadhyaya/PyScripts/export.xml")
root = (tree.getroot())
#print (root)

path = "/Users/snehalr/Documents/Snehal/Swadhyaya/PyScripts"
file_path = path + '/export.xml'

records = [i.attrib for i in root.iter("Record")]
#print (records)

records_df = pd.DataFrame(records)
records_df.drop(['sourceName', 'sourceVersion', 'device'], axis=1, inplace=True)

results = records_df.query('type in ["HKQuantityTypeIdentifierHeartRate", "HKQuantityTypeIdentifierHeartRateVariabilitySDNN","HKCategoryTypeIdentifierSleepAnalysis"]')

results.to_csv(path + '/export.csv', index=False)
