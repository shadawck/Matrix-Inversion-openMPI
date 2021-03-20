## Inversion de matrice n par n avec openMPI

### Prog
```shell
>> ./pp_tp3 10
----------------
Matrix matrixSize : X

----------------
Error : 
[[x,x,x,x,x,x],
 [x,x,x,x,x,x],
 ...
 [x,x,x,x,x,x]]

----------------
Inversion time : X
```

### 2 Steps : 
Forward elimination : Reduces a given system to row echelon form,
Backward elimination : Puts the matrix into reduced row echelon form.  

For n-by-n matrix computation time is : O(n^3)