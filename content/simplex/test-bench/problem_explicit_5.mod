/* Problem 5: maximization */
var x1 >= 0;
var x2 >= 0;

maximize z: 31 * x1 +  21 * x2;

subject to
  c1: 1 * x1 +  21 * x2 >= 6;
  c2: 41 * x1 +  1 * x2 >= 10;
  c3: 1 * x1 +  1 * x2 >= 7;

end;
