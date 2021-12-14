spazio = 0.0
n_layers = 13
n_z = 13
n_y = 13
with open("cellette.tg", "w", encoding = 'utf-8') as file:
    for layers in range(1,n_layers):
        i = 0
        for num_z in range(1,n_z):
            for num_y in range(1,n_y):
                i+=1
                file.write(f":volu cella{layers*1000+i}(S) celletta G4_SODIUM_IODIDE\n")

    for layers in range(1,n_layers):
        i = 0
        for num_z in range(1,n_z):
            for num_y in range(1,n_y):
                i+=1
                file.write(f":place cella{layers*1000+i}(S) {layers*1000+i} world r000 {(5.+spazio)*(layers-1)}*cm {(2.+spazio)*(num_y-1)}*cm {(2.+spazio)*(num_z-1)}*cm\n")

    file.write(":color cella*(S) 1.0 1.0 1.0")
