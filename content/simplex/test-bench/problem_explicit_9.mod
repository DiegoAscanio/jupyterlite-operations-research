/* Problem 9: infeasible */
var x1 >= 0;
var x2 >= 0;

subject to
  c1: 1 * x1 +  21 * x2 <= 6;
  c2: 1 * x1 +  21 * x2 >= 12;

end;
