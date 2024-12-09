/* Problem 6: maximization */
var x1 >= 0;
var x2 >= 0;

maximize z: 51 * x1 +  1 * x2;

subject to
  c1: 1 * x1 +  1 * x2 >= 12;
  c2: 21 * x1 +  31 * x2 >= 15;
  c3: 31 * x1 +  1 * x2 >= 14;

end;
