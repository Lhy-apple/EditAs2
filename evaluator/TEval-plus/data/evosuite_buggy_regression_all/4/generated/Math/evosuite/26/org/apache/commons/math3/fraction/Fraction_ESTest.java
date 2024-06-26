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
      Fraction fraction0 = new Fraction((-2147483645), (-2147483645));
      Fraction fraction1 = fraction0.add(fraction0);
      assertEquals(1, fraction0.getDenominator());
      assertEquals((byte)2, fraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      int int0 = fraction0.getDenominator();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      Fraction fraction1 = fraction0.add(15);
      assertEquals(4, fraction1.getDenominator());
      assertEquals(15.75, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.MINUS_ONE.divide(3115);
      boolean boolean0 = fraction1.equals(fraction0);
      assertEquals((-1), fraction1.getNumerator());
      assertFalse(fraction0.equals((Object)fraction1));
      assertEquals((-0.03210272873194221), fraction1.percentageValue(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.ONE_FIFTH.subtract(fraction0);
      assertEquals(20, fraction1.getDenominator());
      assertEquals((-0.05), fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Fraction fraction0 = new Fraction((-166));
      assertEquals(1, fraction0.getDenominator());
      assertEquals((-166), fraction0.intValue());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      FractionField fractionField0 = fraction0.TWO.getField();
      assertNotNull(fractionField0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Fraction fraction0 = new Fraction(220.3);
      Fraction fraction1 = fraction0.ZERO.subtract(fraction0);
      assertEquals((-22030.0), fraction1.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Fraction fraction0 = new Fraction(4, 1439);
      long long0 = fraction0.longValue();
      assertEquals(0.27797081306462823, fraction0.percentageValue(), 0.01);
      assertEquals(4, fraction0.getNumerator());
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = fraction0.ZERO.divide(fraction0);
      assertEquals(0.0, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Fraction fraction0 = new Fraction(11, 11);
      Fraction fraction1 = fraction0.ZERO.subtract(11);
      assertEquals(1, fraction0.getNumerator());
      assertEquals(1, fraction1.getDenominator());
      assertEquals(1, fraction0.intValue());
      assertEquals((-11.0F), fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      int int0 = fraction0.intValue();
      assertEquals(20.0, fraction0.percentageValue(), 0.01);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      fraction0.hashCode();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      int int0 = fraction0.getNumerator();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Fraction fraction0 = new Fraction(4, 1439);
      float float0 = fraction0.floatValue();
      assertEquals(4, fraction0.getNumerator());
      assertEquals(0.0027797082F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Fraction fraction0 = new Fraction(4, 1439);
      double double0 = fraction0.percentageValue();
      assertEquals(4, fraction0.getNumerator());
      assertEquals(0.27797081306462823, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Fraction fraction0 = new Fraction((-1496.5125599674225), (-1215));
      assertEquals((-149700.0), fraction0.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(4.503599627370496E15, (-914));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 4,503,599,627,370,496 to fraction (4,503,599,627,370,496/1)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) 5, 5);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 5 to fraction (9,223,372,036,854,775,804/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((-2147483645), (-2147483645), (-2147483645));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert -2,147,483,645 to fraction (-9,223,372,034,707,292,162/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(5.936509086967856E-5, 5.936509086967856E-5, (-502));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Unable to convert 0 to fraction after -502 iterations
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
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
        fraction0 = new Fraction((-1115), Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -1,115/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      Fraction fraction1 = fraction0.MINUS_ONE.abs();
      assertEquals(1, fraction1.getNumerator());
      assertEquals(1L, fraction1.longValue());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.ONE_FIFTH.abs();
      assertEquals(0.2F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Fraction fraction0 = new Fraction(221.4282284845596);
      int int0 = fraction0.compareTo(fraction0);
      assertEquals(0, int0);
      assertEquals(221.42822384428223, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.THREE_QUARTERS.multiply(63);
      int int0 = fraction0.compareTo(fraction1);
      assertEquals((-1), int0);
      assertEquals(47.25, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Fraction fraction0 = new Fraction(4, 1439);
      Fraction fraction1 = fraction0.MINUS_ONE.negate();
      int int0 = fraction1.compareTo(fraction0);
      assertEquals(1, fraction1.getNumerator());
      assertEquals(0.0027797082F, fraction0.floatValue(), 0.01F);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      boolean boolean0 = fraction0.equals("3 / 5");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
      fraction0.equals(fraction0);
      assertEquals(1, fraction0.getDenominator());
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.THREE_QUARTERS.multiply(63);
      boolean boolean0 = fraction1.equals(fraction0);
      assertFalse(boolean0);
      assertEquals(189, fraction1.getNumerator());
      assertEquals(4725.0, fraction1.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(3, 3);
      Fraction fraction1 = Fraction.getReducedFraction(3, 3);
      boolean boolean0 = fraction1.equals(fraction0);
      assertTrue(boolean0);
      assertEquals(1.0F, fraction1.floatValue(), 0.01F);
      assertEquals(1, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Fraction fraction0 = new Fraction((double) Integer.MIN_VALUE);
      // Undeclared exception!
      try { 
        fraction0.negate();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/1, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      // Undeclared exception!
      try { 
        fraction0.ONE_THIRD.subtract((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = fraction0.ZERO.add(fraction0);
      assertSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Fraction fraction0 = new Fraction(9.755392680573412E-9);
      Fraction fraction1 = fraction0.TWO_FIFTHS.subtract(fraction0);
      assertEquals(0.0F, fraction0.floatValue(), 0.01F);
      assertEquals(40.0, fraction1.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = Fraction.ONE_FIFTH;
      Fraction fraction2 = fraction0.ONE_FIFTH.subtract(fraction1);
      assertEquals(0.0, fraction2.doubleValue(), 0.01);
      assertEquals(1, fraction2.getDenominator());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Fraction fraction0 = new Fraction((-2147483645), (-2147483603));
      // Undeclared exception!
      try { 
        fraction0.add(fraction0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow, numerator too large after multiply: 4,294,967,290
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      // Undeclared exception!
      try { 
        fraction0.THREE_FIFTHS.multiply((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.THREE_QUARTERS.multiply(fraction0);
      assertEquals((-0.75F), fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.MINUS_ONE.multiply(fraction0);
      assertEquals(0, fraction1.getNumerator());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
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
  public void test43()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = Fraction.ZERO;
      // Undeclared exception!
      try { 
        fraction0.ONE_FIFTH.divide(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // the fraction to divide by must not be zero: 0/1
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(3, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 3/0
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(0, 475);
      assertEquals(0.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction((-2147483645), Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,645/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(Integer.MIN_VALUE, (-2147483645));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-2,147,483,645, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      String string0 = fraction0.THREE_FIFTHS.toString();
      assertEquals("3 / 5", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      String string0 = fraction0.toString();
      assertEquals("0", string0);
  }
}
