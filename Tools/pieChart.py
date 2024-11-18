import matplotlib.pyplot as plt

#Créer un pie chart pour montrer la consommation moyenne
datas = [2463,508]

#On plot le pie chart
plt.pie(datas, labels=['FIR','PPD'])
plt.gca().patches[0].set_color('green')
plt.gca().patches[1].set_color('blue')
plt.legend(loc='lower right', labels=[
    'FIR - {:.1f}%'.format(datas[0] / sum(datas) * 100),
    'PPD - {:.1f}%'.format(datas[1] / sum(datas) * 100)
])
plt.title('Number of logic cells used by the FIR or the PPD.')
#plt.savefig('Tools/pie_chart1.pdf', format='pdf')
plt.show()