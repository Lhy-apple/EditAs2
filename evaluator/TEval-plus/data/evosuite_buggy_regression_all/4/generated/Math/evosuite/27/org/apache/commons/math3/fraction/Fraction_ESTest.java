/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:08:53 GMT 2023
 */

package org.apache.commons.math3.fraction;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.fraction.Fraction;
import org.apache.commons.math3.fraction.FractionField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Fraction_ESTest extends Fraction_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Fraction fraction0 = new Fraction((-2147483645), (-454));
      // Undeclared exception!
      try { 
        fraction0.THREE_QUARTERS.add(fraction0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow, numerator too large after multiply: 4,294,967,971
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-4775), (-4775));
      Fraction fraction1 = fraction0.multiply(963);
      boolean boolean0 = fraction0.equals(fraction1);
      assertFalse(fraction1.equals((Object)fraction0));
      assertEquals(963.0F, fraction1.floatValue(), 0.01F);
      assertFalse(boolean0);
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
      int int0 = fraction0.getDenominator();
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      Fraction fraction1 = fraction0.add(62461023);
      assertEquals(5, fraction1.getDenominator());
      assertEquals(6.24610238E7, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Fraction fraction0 = new Fraction((double) (-2147483646));
      Fraction fraction1 = fraction0.divide((-2147483646));
      assertEquals(1, fraction1.getNumerator());
      assertEquals(1.0, fraction1.doubleValue(), 0.01);
      assertEquals((-2.14748365E9F), fraction0.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      Fraction fraction1 = fraction0.TWO_THIRDS.subtract(fraction0);
      assertEquals(3, fraction1.getDenominator());
      assertEquals((-1.3333333333333333), fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Fraction fraction0 = new Fraction((-1));
      assertEquals((-1), fraction0.getNumerator());
      assertEquals((-1), fraction0.intValue());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      FractionField fractionField0 = fraction0.getField();
      assertNotNull(fractionField0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      long long0 = fraction0.longValue();
      assertEquals(0.8F, fraction0.floatValue(), 0.01F);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      Fraction fraction1 = fraction0.FOUR_FIFTHS.subtract((-1056));
      assertEquals(5, fraction1.getDenominator());
      assertEquals(1056.8F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      int int0 = fraction0.intValue();
      assertEquals(0.5, fraction0.doubleValue(), 0.01);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      fraction0.hashCode();
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_THIRDS;
      int int0 = fraction0.getNumerator();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      float float0 = fraction0.floatValue();
      assertEquals(0.8F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_THIRDS;
      double double0 = fraction0.THREE_FIFTHS.percentageValue();
      assertEquals((byte)0, fraction0.byteValue());
      assertEquals(60.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) (-696), (-696));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert -696 to fraction (697/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(1.633123935319537E16);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 16,331,239,353,195,370 to fraction (16,331,239,353,195,370/1)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) 1959, 1959);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 1,959 to fraction (9,223,372,036,854,773,850/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Fraction fraction0 = new Fraction((-2020.677205258997));
      Fraction fraction1 = fraction0.abs();
      assertEquals(2020.6772151898733, fraction1.doubleValue(), 0.01);
      assertEquals(158, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((-303.1), (-303.1), 2);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Unable to convert -303.1 to fraction after 2 iterations
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Fraction fraction0 = new Fraction(1436.075848957204, 1482);
      assertEquals(1436.0758F, fraction0.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(0, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 0/0
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(2147483631, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 2,147,483,631/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      Fraction fraction1 = fraction0.abs();
      assertEquals(0.4, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      int int0 = fraction0.compareTo(fraction0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      Fraction fraction1 = new Fraction((-2020.677205258997));
      int int0 = fraction1.compareTo(fraction0);
      assertEquals((-2020.6772151898733), fraction1.doubleValue(), 0.01);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      Fraction fraction1 = Fraction.getReducedFraction((-1737), (-1737));
      int int0 = fraction0.compareTo(fraction1);
      assertEquals(1.0F, fraction1.floatValue(), 0.01F);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-4775), (-4775));
      Fraction fraction1 = Fraction.getReducedFraction(3074, 3074);
      boolean boolean0 = fraction1.equals(fraction0);
      assertTrue(boolean0);
      assertEquals(1.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      boolean boolean0 = fraction0.equals(fraction0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      boolean boolean0 = fraction0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      Fraction fraction1 = fraction0.THREE_QUARTERS.add(fraction0);
      boolean boolean0 = fraction1.equals(fraction0);
      assertFalse(fraction0.equals((Object)fraction1));
      assertEquals(1.5F, fraction1.floatValue(), 0.01F);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, 89);
      // Undeclared exception!
      try { 
        fraction0.negate();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/89, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      // Undeclared exception!
      try { 
        fraction0.TWO_THIRDS.subtract((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertNotSame(fraction1, fraction0);
      assertTrue(fraction1.equals((Object)fraction0));
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.ZERO.add(fraction0);
      assertSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.FOUR_FIFTHS.add(fraction0);
      assertEquals(0.8F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      Fraction fraction1 = Fraction.ONE_QUARTER;
      Fraction fraction2 = fraction1.add(fraction0);
      assertEquals(1.05F, fraction2.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_THIRDS;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertEquals(1, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      Fraction fraction1 = fraction0.ONE_QUARTER.divide(fraction0);
      assertEquals(0.625, fraction1.doubleValue(), 0.01);
      assertEquals(5, fraction1.getNumerator());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      // Undeclared exception!
      try { 
        fraction0.multiply((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = fraction0.ZERO.multiply(fraction0);
      assertEquals(0.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.TWO_THIRDS.multiply(fraction1);
      assertEquals(0.0, fraction2.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_THIRD;
      // Undeclared exception!
      try { 
        fraction0.MINUS_ONE.divide((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      Fraction fraction1 = Fraction.ZERO;
      // Undeclared exception!
      try { 
        fraction0.divide(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // the fraction to divide by must not be zero: 0/1
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(0, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 0/0
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(0, (-1));
      assertEquals(0.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
      assertEquals(1, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction((-1), Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -1/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(Integer.MIN_VALUE, (-1));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-1, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      String string0 = fraction0.toString();
      assertEquals("1 / 2", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-4775), (-4775));
      String string0 = fraction0.toString();
      assertEquals("1", string0);
  }
}