import matplotlib.pyplot as plt
import numpy as np


# x = np.array([2023,2024,2025,2026])
# y1 = np.array([15, 25, 30, 20])

# plt.plot(x, y1, marker = ".",
#                markersize = 25,
#                markeredgecolor = "#631813",
#                linestyle = "solid",
#                markerfacecolor = "#631813",
#                linewidth = 3,
#                color = "#cc9a97")


# plt.show()




# x = np.array([2023,2024,2025,2026])
# y1 = np.array([15, 25, 30, 20])
# y2 = np.array([17,23,38,5])


# LineStyle = dict(marker = ".",
#                markersize = 25,
#                markeredgecolor = "#631813",
#                linestyle = "solid",
#                markerfacecolor = "#631813",
#                linewidth = 3,)

# plt.plot(x, y1, color = "#1e30d4", **LineStyle)
# plt.plot(x, y2, color = "#cc9a97", **LineStyle)


# plt.title("Class Size", fontsize = 20,
#                         fontweight = "bold",
#                         family = "Arial",
#                         color = "red")


# # plt.tick_params('x',colors = "#631813")
# plt.tick_params("both",colors = "#631813")

# plt.xlabel("Years", fontsize = 14,
#                     fontweight = "bold",
#                     family = "Arial",
#                     color = "red")
# plt.ylabel("Students", fontsize = 14,
#                     fontweight = "bold",
#                     family = "Arial",
#                     color = "red")


# # plt.xticks(x)


# plt.grid()
# # plt.grid(axis = "y", color = "red")

# plt.show()  








###Bar Chart###

# Catigories = np.array(["Grains", "Fruits", "Vegetables", "Protien", "Dairy", "Sweets"])

# Values = np.array([3,4,2,6,3,1])

# plt.bar(Catigories, Values, color = "skyblue")
# # plt.barh(Catigories, Values, color = "skyblue")
# plt.show()











###Bie Chart###

# categories = ["Freshmen", "Sophomores", "Juniors", "Seniors"]
# values = np.array([300,250,275,225])
# color = ["skyblue","green","blue","red"]

# plt.pie(values, labels= categories,
#                 autopct = "%1.1f%%",
#                 colors= color,
#                 explode = [0,0,0.1,0],
#                 shadow = True,
#                 startangle= 90)

# plt.show()














###Scatter Graphs###
# x = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8])  # Hours studied
# y = np.array([55, 60, 65, 62, 68, 70, 75, 78, 82, 85, 87])  # Grades

# plt.scatter(x, y, color="skyblue", alpha=0.5, s=200)
# plt.show()
















###Histograms###
