/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:38:46 GMT 2023
 */

package org.joda.time.field;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.DateTimeField;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.joda.time.DurationField;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.field.FieldUtils;
import org.joda.time.field.UnsupportedDateTimeField;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FieldUtils_ESTest extends FieldUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      int int0 = FieldUtils.getWrappedValue((-2250), 0, 0, 1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      int int0 = FieldUtils.safeNegate(2910);
      assertEquals((-2910), int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeNegate(Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Integer.MIN_VALUE cannot be negated
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      int int0 = FieldUtils.safeAdd(668, 668);
      assertEquals(1336, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      int int0 = FieldUtils.safeAdd(724, (-2250));
      assertEquals((-1526), int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeAdd(2058005162, 2058005162);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // The calculation caused an overflow: 2058005162 + 2058005162
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      long long0 = FieldUtils.safeAdd(1193L, 1193L);
      assertEquals(2386L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeAdd((-9223372036854775728L), (-9223372036854775728L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // The calculation caused an overflow: -9223372036854775728 + -9223372036854775728
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      long long0 = FieldUtils.safeAdd((-2147483618L), 21260793600000L);
      assertEquals(21258646116382L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      long long0 = FieldUtils.safeSubtract(2147483647L, 2147483647L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeSubtract(557L, (-9223372036854775749L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // The calculation caused an overflow: 557 - -9223372036854775749
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      long long0 = FieldUtils.safeSubtract((-1L), (-1L));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiply((-1978), 1437601);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Multiplication overflows an int: -1978 * 1437601
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      int int0 = FieldUtils.safeMultiply((-1), (-1));
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiply((-2146490354), (-2146490354));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Multiplication overflows an int: -2146490354 * -2146490354
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      long long0 = FieldUtils.safeMultiply((long) 1, (-1));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      long long0 = FieldUtils.safeMultiply(1129L, 1274641);
      assertEquals(1439069689L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      long long0 = FieldUtils.safeMultiply(0L, 0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      long long0 = FieldUtils.safeMultiply((-2676L), 1);
      assertEquals((-2676L), long0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiply(9223372036854775807L, (-2137541593));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Multiplication overflows a long: 9223372036854775807 * -2137541593
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      long long0 = FieldUtils.safeMultiply(1L, 1L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      long long0 = FieldUtils.safeMultiply(1L, (-9223372036854775699L));
      assertEquals((-9223372036854775699L), long0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      int int0 = FieldUtils.safeMultiplyToInt(0L, 0L);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      int int0 = FieldUtils.safeMultiplyToInt(1010L, 0L);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiply(31449600000L, 31449600000L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Multiplication overflows a long: 31449600000 * 31449600000
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiply((-9223372036854775808L), (-1L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Multiplication overflows a long: -9223372036854775808 * -1
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiply((-1L), (-9223372036854775803L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Multiplication overflows a long: -1 * -9223372036854775803
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeMultiplyToInt(469983704766L, (-1354L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Value cannot fit in an int: -636357936253164
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.safeToInt(5097600000L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Value cannot fit in an int: 5097600000
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekyearOfCentury();
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance(dateTimeZone0);
      DurationField durationField0 = buddhistChronology0.eras();
      UnsupportedDateTimeField unsupportedDateTimeField0 = UnsupportedDateTimeField.getInstance(dateTimeFieldType0, durationField0);
      // Undeclared exception!
      try { 
        FieldUtils.verifyValueBounds((DateTimeField) unsupportedDateTimeField0, 1, 2528, (-1316414));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1 for weekyearOfCentury must be in the range [2528,-1316414]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.weekyearOfCentury();
      DateTimeZone dateTimeZone0 = DateTimeZone.getDefault();
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance(dateTimeZone0);
      DurationField durationField0 = buddhistChronology0.eras();
      UnsupportedDateTimeField unsupportedDateTimeField0 = UnsupportedDateTimeField.getInstance(dateTimeFieldType0, durationField0);
      // Undeclared exception!
      try { 
        FieldUtils.verifyValueBounds((DateTimeField) unsupportedDateTimeField0, 1, 1, (-1316414));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1 for weekyearOfCentury must be in the range [1,-1316414]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstanceUTC();
      DateTimeField dateTimeField0 = buddhistChronology0.weekyearOfCentury();
      FieldUtils.verifyValueBounds(dateTimeField0, 1, 1, 1);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      // Undeclared exception!
      try { 
        FieldUtils.verifyValueBounds(dateTimeFieldType0, (-1526), 1023, (-1526));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -1526 for minuteOfDay must be in the range [1023,-1526]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.minuteOfDay();
      FieldUtils.verifyValueBounds(dateTimeFieldType0, 1, 1, 1);
      assertEquals("minuteOfDay", dateTimeFieldType0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      DateTimeFieldType dateTimeFieldType0 = DateTimeFieldType.yearOfCentury();
      // Undeclared exception!
      try { 
        FieldUtils.verifyValueBounds(dateTimeFieldType0, 28181250, 28181250, 246);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 28181250 for yearOfCentury must be in the range [28181250,246]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.verifyValueBounds(" D!`|s(4:gT", 0, 3967, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for  D!`|s(4:gT must be in the range [3967,0]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.verifyValueBounds("}G\"r", 0, 0, (-2861));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 0 for }G\"r must be in the range [0,-2861]
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      FieldUtils.verifyValueBounds("dayOfWeek", 1524, 1524, 1524);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      // Undeclared exception!
      try { 
        FieldUtils.getWrappedValue((-2146490354), (-2146490354), (-2146490354));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // MIN > MAX
         //
         verifyException("org.joda.time.field.FieldUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      int int0 = FieldUtils.getWrappedValue((-2165), (-2146666802), 2165);
      assertEquals((-2165), int0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      int int0 = FieldUtils.getWrappedValue((-2142), (-794), 2142);
      assertEquals(795, int0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Object object0 = new Object();
      boolean boolean0 = FieldUtils.equals((Object) null, object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      boolean boolean0 = FieldUtils.equals((Object) "", (Object) "");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Object object0 = new Object();
      boolean boolean0 = FieldUtils.equals(object0, (Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Object object0 = new Object();
      boolean boolean0 = FieldUtils.equals(object0, (Object) "");
      assertFalse(boolean0);
  }
}
