/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:18:47 GMT 2023
 */

package org.joda.time.field;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.DurationField;
import org.joda.time.DurationFieldType;
import org.joda.time.Minutes;
import org.joda.time.field.DelegatedDurationField;
import org.joda.time.field.UnsupportedDurationField;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UnsupportedDurationField_ESTest extends UnsupportedDurationField_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.seconds();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      assertNotNull(unsupportedDurationField0);
      
      boolean boolean0 = unsupportedDurationField0.isSupported();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance((DurationFieldType) null);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getMillis((-372), (long) (-372));
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // null field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.years();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.subtract(0L, 0L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // years field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.millis();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.subtract(11L, 1567);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // millis field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.weekyears();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getDifferenceAsLong((-636L), (-636L));
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // weekyears field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.weeks();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getValue(1515L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // weeks field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.hours();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getValueAsLong(0L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // hours field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.millis();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getMillis((-3831L), (-3831L));
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // millis field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.eras();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      int int0 = unsupportedDurationField0.compareTo((DurationField) unsupportedDurationField0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Minutes minutes0 = Minutes.THREE;
      DurationFieldType durationFieldType0 = minutes0.getFieldType();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      unsupportedDurationField0.hashCode();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.seconds();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      long long0 = unsupportedDurationField0.getUnitMillis();
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance((DurationFieldType) null);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getDifference(3283L, 3283L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // null field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.eras();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      DelegatedDurationField delegatedDurationField0 = new DelegatedDurationField(unsupportedDurationField0);
      assertEquals("eras", delegatedDurationField0.getName());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.hours();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getValue(957L, 957L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // hours field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.eras();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      boolean boolean0 = unsupportedDurationField0.isPrecise();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.eras();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getMillis(548L);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // eras field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.months();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getMillis(0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // months field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.years();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      // Undeclared exception!
      try { 
        unsupportedDurationField0.getValueAsLong((-1L), (-1L));
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // years field is unsupported
         //
         verifyException("org.joda.time.field.UnsupportedDurationField", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.millis();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      String string0 = unsupportedDurationField0.toString();
      assertEquals("UnsupportedDurationField[millis]", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.seconds();
      UnsupportedDurationField.getInstance(durationFieldType0);
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      assertFalse(unsupportedDurationField0.isSupported());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.seconds();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      DurationFieldType durationFieldType1 = DurationFieldType.centuries();
      UnsupportedDurationField unsupportedDurationField1 = UnsupportedDurationField.getInstance(durationFieldType1);
      boolean boolean0 = unsupportedDurationField0.equals(unsupportedDurationField1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.seconds();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      boolean boolean0 = unsupportedDurationField0.equals(unsupportedDurationField0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DurationFieldType durationFieldType0 = DurationFieldType.months();
      UnsupportedDurationField unsupportedDurationField0 = UnsupportedDurationField.getInstance(durationFieldType0);
      boolean boolean0 = unsupportedDurationField0.equals(durationFieldType0);
      assertFalse(boolean0);
  }
}