/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:51:42 GMT 2023
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
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = fraction0.THREE_FIFTHS.add(fraction0);
      assertEquals(1.6F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Fraction fraction0 = new Fraction(1.9987994582857286E-8, 1);
      Fraction fraction1 = fraction0.multiply(1);
      boolean boolean0 = fraction1.equals(fraction0);
      assertTrue(boolean0);
      assertEquals(0, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      int int0 = fraction0.getDenominator();
      assertEquals(4, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = fraction0.divide(1829);
      assertEquals(0.054674685620557675, fraction1.percentageValue(), 0.01);
      assertEquals(1, fraction1.getNumerator());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.ONE.subtract(fraction0);
      assertEquals(0.75F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Fraction fraction0 = new Fraction((-214));
      Fraction fraction1 = Fraction.TWO_FIFTHS;
      boolean boolean0 = fraction0.equals(fraction1);
      assertFalse(boolean0);
      assertEquals(1, fraction0.getDenominator());
      assertEquals((-214L), fraction0.longValue());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      FractionField fractionField0 = fraction0.getField();
      assertNotNull(fractionField0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((-1589), (-1589), (-1589));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert -1,589 to fraction (-9,223,372,036,854,774,218/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      long long0 = fraction0.longValue();
      assertEquals(20.0, fraction0.percentageValue(), 0.01);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(1.8014398509481984E16);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 18,014,398,509,481,984 to fraction (18,014,398,509,481,984/1)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(0, Integer.MIN_VALUE);
      Fraction fraction1 = fraction0.subtract(Integer.MIN_VALUE);
      // Undeclared exception!
      try { 
        fraction1.negate();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/1, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      int int0 = fraction0.intValue();
      assertEquals(0.25F, fraction0.floatValue(), 0.01F);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      fraction0.hashCode();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      int int0 = fraction0.getNumerator();
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      float float0 = fraction0.floatValue();
      assertEquals(0.4F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-2147483646), (-2147483646));
      double double0 = fraction0.percentageValue();
      assertEquals(100.0, double0, 0.01);
      assertEquals(1, fraction0.getDenominator());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Fraction fraction0 = new Fraction((-510.0));
      assertEquals(1, fraction0.getDenominator());
      assertEquals((byte)2, fraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(1633.0, 1501);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 1,633 to fraction (9,223,372,036,854,774,176/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(1858.36141831036, (-1589), (-1589));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Unable to convert 1,858.361 to fraction after -1,589 iterations
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Fraction fraction0 = new Fraction((-509.910791537));
      Object object0 = new Object();
      boolean boolean0 = fraction0.equals(object0);
      assertEquals((-108611), fraction0.getNumerator());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
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
  public void test21()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(0, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 0/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(Integer.MIN_VALUE, (-444));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-444, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Fraction fraction0 = new Fraction((-214), (-214));
      assertEquals(1, fraction0.getNumerator());
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      Fraction fraction1 = fraction0.MINUS_ONE.abs();
      assertEquals((byte)1, fraction1.byteValue());
      assertEquals(1, fraction1.getNumerator());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = fraction0.abs();
      assertEquals(1.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      int int0 = fraction0.compareTo(fraction0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = Fraction.ONE;
      int int0 = fraction0.compareTo(fraction1);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.divide(fraction0);
      int int0 = fraction1.compareTo(fraction0);
      assertEquals(1, int0);
      assertEquals(1.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      boolean boolean0 = fraction0.equals(fraction0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = Fraction.ONE_QUARTER;
      boolean boolean0 = fraction1.equals(fraction0);
      assertFalse(boolean0);
      assertFalse(fraction0.equals((Object)fraction1));
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      // Undeclared exception!
      try { 
        fraction0.ONE_HALF.add((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      Fraction fraction1 = fraction0.ZERO.subtract(fraction0);
      assertEquals(2, fraction1.getDenominator());
      assertEquals((-50.0), fraction1.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.add(fraction0);
      assertSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-2147483646), (-2147483646));
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.TWO.subtract(fraction1);
      assertEquals(100.0, fraction0.percentageValue(), 0.01);
      assertEquals(1, fraction0.getDenominator());
      assertEquals(2, fraction2.intValue());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = Fraction.TWO_FIFTHS;
      Fraction fraction2 = fraction0.THREE_FIFTHS.add(fraction1);
      assertTrue(fraction2.equals((Object)fraction0));
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = Fraction.THREE_FIFTHS;
      Fraction fraction2 = fraction0.FOUR_FIFTHS.subtract(fraction1);
      assertEquals(0.2F, fraction2.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-2147483647), 2529);
      Fraction fraction1 = fraction0.add((-2147483647));
      // Undeclared exception!
      try { 
        fraction0.subtract(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow, numerator too large after multiply: -2,147,486,177
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      // Undeclared exception!
      try { 
        fraction0.ONE.multiply((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Fraction fraction0 = new Fraction(1.9987994582857286E-8, 1);
      assertEquals((short)0, fraction0.shortValue());
      
      Fraction fraction1 = fraction0.multiply(fraction0);
      assertEquals(1, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.TWO_THIRDS.multiply(fraction1);
      assertEquals(0.0F, fraction2.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      // Undeclared exception!
      try { 
        fraction0.THREE_QUARTERS.divide((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = Fraction.ZERO;
      // Undeclared exception!
      try { 
        fraction0.TWO_QUARTERS.divide(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // the fraction to divide by must not be zero: 0/1
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
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
  public void test44()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
      assertEquals(1, fraction0.getDenominator());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(1429, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 1,429/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(Integer.MIN_VALUE, (-75));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-75, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      String string0 = fraction0.toString();
      assertEquals("1 / 4", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      String string0 = fraction0.toString();
      assertEquals("1", string0);
  }
}