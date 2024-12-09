/* Problem 7: infeasible */
var x1 >= 0;
var x2 >= 0;

subject to
  c1: 1 * x1 +  1 * x2 <= 10;
  c2: 1 * x1 +  1 * x2 >= 20;

end;
