/* Problem 11: mixed */
var x1 >= 0;
var x2 >= 0;

subject to
  c1: 1 * x1 +  1 * x2 >= 5;
  c2: 1 * x1 +  21 * x2 <= 12;
  c3: 1 * x1 +  1 * x2 = 7;

end;
