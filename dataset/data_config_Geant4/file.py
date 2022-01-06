#Geometry parameters
spazio = 0.0 #cm
n_layers = 13
n_z = 13
n_y = 13
X_DIM = 5. #cm
Y_DIM = 2. #cm
Z_DIM = 2. #cm

#write to file positioning and color information

with open("cellette.tg", "w", encoding = 'utf-8') as file:
    for layers in range(1,n_layers):
        i = 0
        for num_z in range(1,n_z):
            for num_y in range(1,n_y):
                i+=1
                file.write(f":volu cella{layers*1000+i}(S) celletta G4_CESIUM_IODIDE\n")

    for layers in range(1,n_layers):
        i = 0
        for num_z in range(1,n_z):
            for num_y in range(1,n_y):
                i+=1
                file.write(f":place cella{layers*1000+i}(S) {layers*1000+i} world r000 {(X_DIM +spazio)*(layers-1)}*cm {(Y_DIM +spazio)*(num_y-1)}*cm {(Z_DIM +spazio)*(num_z-1)}*cm\n")

    file.write(":color cella*(S) 1.0 1.0 1.0")
