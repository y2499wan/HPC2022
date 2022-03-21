hw2: val_test01_solved.cpp
	g++ -std=c++11 -O3 val_test01_solved.cpp -o val_test01_solved.o
	g++ -std=c++11 -O3 val_test02_solved.cpp -o val_test02_solved.o
	g++ -std=c++11 -fopenmp -O3 -march=native MMult1.cpp -o MMult1.o
	g++ -std=c++11 -fopenmp -O3 omp_solved2.c -o omp_solved2.o
	g++ -std=c++11 -fopenmp -O3 omp_solved3.c -o omp_solved3.o
	g++ -std=c++11 -fopenmp -O3 omp_solved4.c -o omp_solved4.o
	g++ -std=c++11 -fopenmp -O3 omp_solved5.c -o omp_solved5.o
	g++ -std=c++11 -fopenmp -O3 omp_solved6.c -o omp_solved6.o
	g++ -std=c++11 -fopenmp -O3 -march=native jacobi2D-omp.cpp -o jacobi2D-omp.o
	g++ -std=c++11 -fopenmp -O3 -march=native gs2D-omp.cpp -o gs2D-omp.o


clean: 
	rm -f *.o