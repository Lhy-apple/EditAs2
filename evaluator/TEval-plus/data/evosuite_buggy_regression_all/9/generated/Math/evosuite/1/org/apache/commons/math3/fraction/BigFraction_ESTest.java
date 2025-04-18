/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:34:42 GMT 2023
 */

package org.apache.commons.math3.fraction;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import org.apache.commons.math3.fraction.BigFraction;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BigFraction_ESTest extends BigFraction_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      BigFraction bigFraction1 = bigFraction0.ONE.divide(bigFraction0);
      assertEquals((short)21496, bigFraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      int int0 = bigFraction0.intValue();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.FOUR_FIFTHS;
      BigInteger bigInteger0 = bigFraction0.getNumerator();
      assertEquals((short)4, bigInteger0.shortValue());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(1246);
      assertEquals((byte) (-34), bigFraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      BigFraction bigFraction0 = new BigFraction(bigInteger0);
      BigInteger bigInteger1 = bigFraction0.getDenominator();
      assertEquals((byte)1, bigInteger1.byteValue());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      bigFraction0.getField();
      assertEquals((byte)0, bigFraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(0L);
      BigFraction bigFraction1 = bigFraction0.ONE_HALF.subtract(0L);
      assertFalse(bigFraction1.equals((Object)bigFraction0));
      assertEquals((short)0, bigFraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ZERO;
      int int0 = bigFraction0.getNumeratorAsInt();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      BigFraction bigFraction1 = bigFraction0.ZERO.divide((-714));
      assertFalse(bigFraction1.equals((Object)bigFraction0));
      assertEquals((short)0, bigFraction0.shortValue());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_QUARTERS;
      BigFraction bigFraction1 = bigFraction0.MINUS_ONE.add((-479));
      assertEquals((byte)32, bigFraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction((double) (-1233));
      int int0 = bigFraction0.compareTo(bigFraction0);
      assertEquals(0, int0);
      assertEquals((short) (-1233), bigFraction0.shortValue());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      BigFraction bigFraction1 = bigFraction0.add((-1538L));
      float float0 = bigFraction1.floatValue();
      assertEquals((byte) (-1), bigFraction1.byteValue());
      assertEquals((-1538.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(1408, 1408);
      bigFraction0.bigDecimalValue();
      assertEquals((byte)1, bigFraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      double double0 = bigFraction0.percentageValue();
      assertEquals(4.845915307093532E-75, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.getReducedFraction((-1), (-1));
      bigFraction0.FOUR_FIFTHS.getDenominatorAsLong();
      assertEquals((short)1, bigFraction0.shortValue());
      assertEquals((byte)1, bigFraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(1.1067527339037042E-8);
      bigFraction0.TWO_THIRDS.hashCode();
      assertEquals((short)0, bigFraction0.shortValue());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ZERO;
      BigFraction bigFraction1 = bigFraction0.ONE_FIFTH.subtract((-300));
      assertEquals((byte)44, bigFraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE_QUARTER;
      long long0 = bigFraction0.FOUR_FIFTHS.longValue();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.MINUS_ONE;
      BigFraction bigFraction1 = bigFraction0.FOUR_FIFTHS.multiply(1540);
      assertEquals((byte) (-48), bigFraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.TWO_FIFTHS;
      // Undeclared exception!
      try { 
        bigFraction0.TWO_THIRDS.bigDecimalValue(3166, 2147483017);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid rounding mode
         //
         verifyException("java.math.BigDecimal", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ZERO;
      // Undeclared exception!
      try { 
        bigFraction0.TWO.bigDecimalValue((-5692));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid rounding mode
         //
         verifyException("java.math.BigDecimal", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_QUARTERS;
      BigFraction bigFraction1 = bigFraction0.multiply((-293L));
      assertEquals((byte)37, bigFraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ZERO;
      int int0 = bigFraction0.TWO_THIRDS.getDenominatorAsInt();
      assertEquals(3, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction((-449.0201654), (-459));
      assertEquals((short) (-450), bigFraction0.shortValue());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_FIFTHS;
      BigFraction bigFraction1 = bigFraction0.divide((-1764L));
      assertEquals((byte)0, bigFraction1.byteValue());
      assertFalse(bigFraction1.equals((Object)bigFraction0));
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      long long0 = bigFraction0.TWO_FIFTHS.getNumeratorAsLong();
      assertEquals(2L, long0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ZERO;
      double double0 = bigFraction0.TWO_FIFTHS.pow(0.1391000000000986);
      assertEquals(0.8803321136757711, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BigFraction bigFraction0 = null;
      try {
        bigFraction0 = new BigFraction(1.7976931348623157E308, 2.0902938842773438, 1033);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 179,769,313,486,231,570,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000 to fraction (9,223,372,036,854,775,807/1)
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(202.06676, 202.06676, 1251);
      assertEquals((byte) (-54), bigFraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BigFraction bigFraction0 = null;
      try {
        bigFraction0 = new BigFraction(1212.6391, (-4341.2236701), 1033);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 1,212.639 to fraction (1,229,777,678,853/1,014,133,289)
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BigFraction bigFraction0 = null;
      try {
        bigFraction0 = new BigFraction(4.8459153070935316E-77, (-1054));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 0 to fraction (1/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BigFraction bigFraction0 = null;
      try {
        bigFraction0 = new BigFraction((-182.1278411347411), (-182.1278411347411), (-1233));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Unable to convert -182.128 to fraction after -1,233 iterations
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.getReducedFraction(0, 796);
      // Undeclared exception!
      try { 
        bigFraction0.add((BigFraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.getReducedFraction((-1), 1);
      BigFraction bigFraction1 = bigFraction0.abs();
      assertNotSame(bigFraction1, bigFraction0);
      assertEquals((byte) (-1), bigFraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE;
      BigFraction bigFraction1 = bigFraction0.abs();
      assertEquals((short)1, bigFraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.TWO;
      BigFraction bigFraction1 = bigFraction0.ONE_HALF.add(bigFraction0);
      assertFalse(bigFraction1.equals((Object)bigFraction0));
      assertEquals((byte)2, bigFraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(0L);
      BigFraction bigFraction1 = bigFraction0.add(bigFraction0);
      assertSame(bigFraction1, bigFraction0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_QUARTERS;
      BigFraction bigFraction1 = bigFraction0.add(bigFraction0);
      assertEquals((byte)1, bigFraction1.byteValue());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      // Undeclared exception!
      try { 
        bigFraction0.TWO_QUARTERS.divide((BigInteger) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(0.02756666666664387, 0.02756666666664387, 121);
      BigInteger bigInteger0 = BigInteger.ZERO;
      // Undeclared exception!
      try { 
        bigFraction0.ZERO.divide(bigInteger0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // denominator must be different from 0
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      // Undeclared exception!
      try { 
        bigFraction0.divide((BigFraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(2147483647L, 2147483647L);
      BigFraction bigFraction1 = new BigFraction(0, (-1L));
      // Undeclared exception!
      try { 
        bigFraction0.divide(bigFraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // denominator must be different from 0
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE;
      boolean boolean0 = bigFraction0.equals(bigFraction0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      boolean boolean0 = bigFraction0.ONE_HALF.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(885, (-86));
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)6;
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigFraction bigFraction1 = new BigFraction(bigInteger0, bigInteger0);
      boolean boolean0 = bigFraction0.ONE_FIFTH.equals(bigFraction1);
      assertFalse(boolean0);
      assertEquals((short) (-10), bigFraction0.shortValue());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(0.028, 0.028, 109);
      float float0 = bigFraction0.floatValue();
      assertEquals(0.028571429F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      // Undeclared exception!
      try { 
        bigFraction0.multiply((BigInteger) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // null is not allowed
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE_THIRD;
      // Undeclared exception!
      try { 
        bigFraction0.multiply((BigFraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE;
      BigInteger bigInteger0 = BigInteger.ONE;
      BigFraction bigFraction1 = bigFraction0.subtract(bigInteger0);
      BigFraction bigFraction2 = bigFraction1.multiply(bigFraction0);
      assertTrue(bigFraction2.equals((Object)bigFraction1));
      assertFalse(bigFraction2.equals((Object)bigFraction0));
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE;
      BigInteger bigInteger0 = BigInteger.ONE;
      BigFraction bigFraction1 = bigFraction0.subtract(bigInteger0);
      BigFraction bigFraction2 = bigFraction0.multiply(bigFraction1);
      assertTrue(bigFraction2.equals((Object)bigFraction1));
      assertFalse(bigFraction2.equals((Object)bigFraction0));
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction((-657.6));
      BigFraction bigFraction1 = bigFraction0.pow(2682);
      assertEquals((short)12508, bigFraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_QUARTERS;
      // Undeclared exception!
      try { 
        bigFraction0.ZERO.pow((-1635));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // denominator must be different from 0
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.MINUS_ONE;
      bigFraction0.pow((long) 1);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_QUARTERS;
      bigFraction0.FOUR_FIFTHS.pow((-2477L));
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      BigFraction bigFraction0 = BigFraction.TWO;
      bigFraction0.pow(bigInteger0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_FIFTHS;
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte) (-40);
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      bigFraction0.pow(bigInteger0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(0.02756666666664387, 0.02756666666664387, 121);
      // Undeclared exception!
      try { 
        bigFraction0.ONE.subtract((BigInteger) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // null is not allowed
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction((double) (-1233));
      bigFraction0.subtract(bigFraction0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.ONE;
      // Undeclared exception!
      try { 
        bigFraction0.TWO_FIFTHS.subtract((BigFraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.BigFraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction((double) (-1233));
      BigFraction bigFraction1 = new BigFraction(0L, (long) (-1233));
      bigFraction0.subtract(bigFraction1);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      BigFraction bigFraction0 = BigFraction.THREE_QUARTERS;
      bigFraction0.TWO.subtract(bigFraction0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(4.8459153070935316E-77);
      bigFraction0.toString();
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte) (-59);
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      BigFraction bigFraction0 = new BigFraction(bigInteger0, bigInteger0);
      bigFraction0.toString();
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      BigFraction bigFraction0 = new BigFraction(0.0);
      bigFraction0.toString();
  }
}
