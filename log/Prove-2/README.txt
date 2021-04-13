-Chi

	Prove fatte salvando i chis per ogni iterazione di tensorflow

	Parametri utlizzati: 
						classi = [Setosa, Versicolor, Virginica]
						c = [0.05,1,75,200]
						sigma = [0.1,0.25,0.5]
						penalization = [0.1,10,100]
-N_Iter_3000

	Prove fatte con un numero di iterazioni maggiore per vedere l'andamento

	Parametri utlizzati: 
						classi = [Setosa, Versicolor, Virginica]
						c = [1,75,200]
						sigma = [0.1,0.25,0.5]
						penalization = [0.1]
-Low-Penalization

	Notando che con penalizzazione bassa avevo risultati migliori ho effettuato prove con valori bassi di penalization

	Parametri utlizzati: 
						classi = [Setosa, Versicolor, Virginica]
						c = [1,75,200]
						sigma = [0.1,0.25,0.5]
						penalization = [0.1,0.01,0.001]

-Prove penalization 01
	
	Notando risultati miglior con penalization sulla scala 0.1, ho provato con i seguenti parametri
	
		Parametri utlizzati: 
						classi = [Setosa, Versicolor, Virginica]
						c = [1,75,200]
						sigma = [0.1,0.25,0.5]
						penalization = [0.1,0.5,1]

-Learning Rate
	
	Prove fatte modficando i valori di learning rate, dell' ottimizzatore Adam di tensorflow
	
		Parametri utlizzati: 
						classi = [Setosa, Versicolor, Virginica]
						c = [1,75,200]
						sigma = [0.1,0.25,0.5]
						penalization = [0.1,0.5,1]
						learning rate = [1e-3,1e-5]
Non tutti gli esperimenti fatti solo i casi più oscillanti che avevo trovato prima
