#Common
import os

#For reading
from lxml import etree
from lxml.etree import XMLParser
import re

#For writing
import csv



sourceDir = "/home/sounak/Resources/Data/ACE Dataset/ace_2005_td_v7/data/English/nw/fp2/"
outDir = "/home/sounak/Resources/Data/ACE Dataset/ace_2005_td_v7/data/English/nw/fp2_apf_extracted/"


for path, subdirs, files in os.walk(sourceDir):
	for name in files:
		if name.endswith("apf.xml"):
			parser = XMLParser(recover=True, encoding="utf-8")
			tree = etree.parse(os.path.join(path, name), parser=parser)
			root = tree.getroot()
			with open(outDir+name.replace('apf.xml', 'csv'), 'w') as outFile:
				writer = csv.writer(outFile, delimiter ='\t')
				#print("Fileame : "+ name)

				sent = ""
				trig = ""
				trigType  = ""
				for doc in root.findall('document'):
					for ev in doc.findall('event'):
						trigType = ("|").join([ev.attrib['TYPE'], ev.attrib['SUBTYPE']])
						for evm in ev.findall('event_mention'):
							sent = evm.find("ldc_scope/charseq").text.replace('\n',' ').replace('\t',' ').replace('\"','')
							#sent = evm.find("extent/charseq").text.replace('\n',' ').replace('\t',' ').replace('\"','')
							trig = evm.find("anchor/charseq").text.replace('\n',' ').replace('\t',' ').replace('\"','')
							arguments = []
							for evma in evm.findall('event_mention_argument'):
								arguments.append("|".join([evma.find("extent/charseq").text.replace('\n',' ').replace('\t',' ').replace('\"',''), evma.attrib["ROLE"]]))

							#if trig == 'sickened w':
								#print(name)

							#Write to file
							out = []
							if len(sent) < 2:
								sent = '-'
							out.append(sent)
							if len(trig) < 2:
								trig = '-'
							out.append(trig)
							if len(trigType) < 2:
								trigType = '-'
							out.append(trigType)
							if len(arguments) < 2:
								arguments = '-'
							out.extend(arguments)
							writer.writerow(out)
					#print("#############################################")
