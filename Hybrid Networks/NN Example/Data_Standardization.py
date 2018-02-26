#Progrm to perform data transformation to directly feed into the neural network
#This prgram only creates models for trigger detection

def oneHot_to_standard(doc_vectors, trig_vectors, tags, sent_len=11):

	if sent_len%2 == 0:
		raise ValueError("Please enter a  odd number as windows size, to ensure even distribution around target word.")
	doc_split, trig_split, tag_split = split(doc_vectors, trig_vectors, tags, sent_len)
	ret_doc = []
	ret_trig = []
	if not len(trig_split) == len(doc_split):
		raise IndexError("Length of sentence splits for words and triger labels do not match!!!")
	for doc, trig in zip(doc_split, trig_split):
		ret_doc.append(doc)
		if trig[int(sent_len/2)] == 1:
			ret_trig.append([1,0])
		else:
			ret_trig.append([0,1])

	return (ret_doc, ret_trig, tag_split)




def split(doc_vectors, trig_vectors, tags, sent_len):

	"""
	#If padding is not done earlier
	while len(doc_vectors) < sent_len:
		doc_vectors.append(0)
		trig_vectors.append(0)
		tags.append("Empty")
		print("Entering padding in Data_Standardization :: split")
	"""

	doc_split = []
	trig_split = []
	tag_split = []

	doc_len = len(doc_vectors)
	for i in range(doc_len):

		curr_doc_window = []
		curr_trig_window = []
		curr_tag_window = []
		window_fill_index = 0

		#Padding to shift initial words to the centre
		while window_fill_index + i < int(sent_len/2):
			curr_doc_window.append(0)
			curr_trig_window.append(0)
			curr_tag_window.append(".")
			window_fill_index += 1

		#Fill with sentence content
		#!Get Start position in sentence
		sent_index = i - int(sent_len/2)
		if sent_index < 0:
			sent_index = 0
		#!Repat till eithe sentence runs out of words or window is filled
		while sent_index < len(doc_vectors) and window_fill_index < sent_len:
			curr_doc_window.append(doc_vectors[sent_index])
			curr_trig_window.append(trig_vectors[sent_index])
			curr_tag_window.append(tags[sent_index])
			window_fill_index += 1
			sent_index += 1

		#Padding if last few words are at the centre
		while window_fill_index < sent_len:
			curr_doc_window.append(0)
			curr_trig_window.append(0)
			curr_tag_window.append(".")
			window_fill_index += 1

		"""
		print("Data_Standardization :: split")
		print("Lengths ; doc, trigger, tag :", len(curr_doc_window), len(curr_trig_window), len(curr_tag_window))
		print(curr_doc_window)
		print("Data_Standardization :: split")
		"""

		doc_split.append(curr_doc_window)
		trig_split.append(curr_trig_window)
		tag_split.append(curr_tag_window)

	return (doc_split, trig_split, tag_split)
