/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 07:03:49 GMT 2023
 */

package org.joda.time.field;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.joda.time.DateTimeField;
import org.joda.time.DateTimeZone;
import org.joda.time.chrono.BuddhistChronology;
import org.joda.time.chrono.GJChronology;
import org.joda.time.field.LenientDateTimeField;
import org.joda.time.field.StrictDateTimeField;
import org.joda.time.tz.FixedDateTimeZone;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LenientDateTimeField_ESTest extends LenientDateTimeField_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance((DateTimeZone) null);
      DateTimeField dateTimeField0 = buddhistChronology0.weekyearOfCentury();
      StrictDateTimeField strictDateTimeField0 = new StrictDateTimeField(dateTimeField0);
      GJChronology gJChronology0 = GJChronology.getInstance();
      DateTimeField dateTimeField1 = LenientDateTimeField.getInstance(strictDateTimeField0, gJChronology0);
      assertNotNull(dateTimeField1);
      
      DateTimeField dateTimeField2 = LenientDateTimeField.getInstance(dateTimeField1, gJChronology0);
      assertTrue(dateTimeField2.isLenient());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      FixedDateTimeZone fixedDateTimeZone0 = (FixedDateTimeZone)DateTimeZone.UTC;
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance((DateTimeZone) fixedDateTimeZone0);
      DateTimeField dateTimeField0 = buddhistChronology0.weekyear();
      LenientDateTimeField lenientDateTimeField0 = new LenientDateTimeField(dateTimeField0, buddhistChronology0);
      long long0 = lenientDateTimeField0.set(1092L, 1);
      assertEquals((-79239686398908L), long0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BuddhistChronology buddhistChronology0 = BuddhistChronology.getInstance();
      DateTimeField dateTimeField0 = LenientDateTimeField.getInstance((DateTimeField) null, buddhistChronology0);
      assertNull(dateTimeField0);
  }
}