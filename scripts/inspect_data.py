import csv
import matplotlib.pyplot as plt 
import numpy as np

def extract_data(path):
	cats = [0, 0, 0, 0, 0, 0, 0]
	csvr = csv.reader(open(path))
	header =  next(csvr)
	
	for row in csvr:
		cats[int(row[0])] += 1

	return cats

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def plot_categories(categories):
	N = 7 
	ind = np.arange(N)
	
	c = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Netural']
	fig, ax = plt.subplots()
	graph = ax.bar(ind+.5, categories, width=0.75)
	graph[0].set_color('aqua')
	graph[1].set_color('blueviolet')
	graph[2].set_color('crimson')
	graph[3].set_color('darksalmon')
	graph[4].set_color('g')
	graph[5].set_color('m')

	ax.set_title('Distribution of Classes in the Testing Set')
	ax.set_ylim(0, max(categories)+700)
	ax.set_ylabel('Count')
	ax.set_xlabel('Classes of Emotions')
	ax.set_xticks(ind+.5 + 0.35)
	ax.set_xticklabels(c)

	for g in graph:
		height = g.get_height()
		ax.text(g.get_x() + g.get_width()/2., 1.05*height, 
			'%d' % int(height), ha='center', va='bottom')

	plt.show()


def main():
	train_cats = extract_data('../data/fer2013/train.csv')
	test_cats = extract_data('../data/fer2013/test.csv')
	
	plot_categories(test_cats)

if __name__ == '__main__':
	main()