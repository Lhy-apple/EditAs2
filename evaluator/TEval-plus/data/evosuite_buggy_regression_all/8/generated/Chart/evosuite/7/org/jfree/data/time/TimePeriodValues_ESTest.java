/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:51:01 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.time.Instant;
import java.util.Date;
import javax.swing.JLayeredPane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.time.MockInstant;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.Day;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Hour;
import org.jfree.data.time.Millisecond;
import org.jfree.data.time.Minute;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.SpreadsheetDate;
import org.jfree.data.time.TimePeriod;
import org.jfree.data.time.TimePeriodValue;
import org.jfree.data.time.TimePeriodValues;
import org.jfree.data.time.Week;
import org.jfree.data.time.Year;
import org.jfree.data.xy.XYDataItem;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TimePeriodValues_ESTest extends TimePeriodValues_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      int int0 = timePeriodValues0.getMaxMiddleIndex();
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Byte byte0 = new Byte((byte)5);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(byte0, "Null 'date' argument.", "Null 'date' argument.");
      timePeriodValues0.setDomainDescription("Null 'date' argument.");
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals("Null 'date' argument.", timePeriodValues0.getDomainDescription());
      assertEquals("Null 'date' argument.", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      timePeriodValues0.add((TimePeriod) week0, (Number) 53);
      Object object0 = timePeriodValues0.clone();
      boolean boolean0 = timePeriodValues0.equals(object0);
      assertEquals(0, timePeriodValues0.getMaxStartIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Instant instant0 = MockInstant.now();
      Date date0 = Date.from(instant0);
      Millisecond millisecond0 = new Millisecond(date0);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(millisecond0);
      // Undeclared exception!
      try { 
        timePeriodValues0.getValue(0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Instant instant0 = MockInstant.now();
      Date date0 = Date.from(instant0);
      Millisecond millisecond0 = new Millisecond(date0);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(millisecond0, "org.jfree.data.time.TimePeriodValues", "Range");
      int int0 = timePeriodValues0.getMinEndIndex();
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals("org.jfree.data.time.TimePeriodValues", timePeriodValues0.getDomainDescription());
      assertEquals("Range", timePeriodValues0.getRangeDescription());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Instant instant0 = MockInstant.now();
      Date date0 = Date.from(instant0);
      Millisecond millisecond0 = new Millisecond(date0);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(millisecond0, "org.jfree.data.time.TimePeriodValues", "Range");
      int int0 = timePeriodValues0.getMaxStartIndex();
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals("org.jfree.data.time.TimePeriodValues", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals("Range", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Nul 'dat' argument.", "Nul 'dat' argument.", "Nul 'dat' argument.");
      Minute minute0 = new Minute();
      timePeriodValues0.add((TimePeriod) minute0, 2632.109);
      assertEquals(0, timePeriodValues0.getMinMiddleIndex());
      assertEquals(0, timePeriodValues0.getMinStartIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockDate mockDate0 = new MockDate(29, 29, 29, 29, 29);
      Day day0 = new Day(mockDate0);
      Hour hour0 = new Hour(29, day0);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(hour0, "D{/+~", "8`8R%Yfv$");
      int int0 = timePeriodValues0.getMinStartIndex();
      assertEquals((-1), int0);
      assertEquals("8`8R%Yfv$", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("D{/+~", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Instant instant0 = MockInstant.now();
      Date date0 = Date.from(instant0);
      Millisecond millisecond0 = new Millisecond(date0);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(millisecond0, "org.jfree.data.time.TimePeriodValues", "Range");
      // Undeclared exception!
      try { 
        timePeriodValues0.getTimePeriod(0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Instant instant0 = MockInstant.now();
      Date date0 = Date.from(instant0);
      Millisecond millisecond0 = new Millisecond(date0);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(millisecond0, "org.jfree.data.time.TimePeriodValues", "Range");
      int int0 = timePeriodValues0.getMaxEndIndex();
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("org.jfree.data.time.TimePeriodValues", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals("Range", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Byte byte0 = new Byte((byte) (-16));
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(byte0, "EQDO6Jt\"JAU=}6/[", "EQDO6Jt\"JAU=}6/[");
      // Undeclared exception!
      try { 
        timePeriodValues0.update((byte) (-16), byte0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      XYDataItem xYDataItem0 = new XYDataItem((-150.397745), (-150.397745));
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(xYDataItem0);
      int int0 = timePeriodValues0.getMinMiddleIndex();
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), int0);
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Null 'date' argument.");
      // Undeclared exception!
      try { 
        timePeriodValues0.add((TimePeriodValue) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null item not allowed.
         //
         verifyException("org.jfree.data.time.TimePeriodValues", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      timePeriodValues0.add((TimePeriod) week0, (Number) 53);
      SpreadsheetDate spreadsheetDate0 = new SpreadsheetDate(400);
      Date date0 = spreadsheetDate0.toDate();
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond(date0);
      RegularTimePeriod regularTimePeriod0 = fixedMillisecond0.previous();
      timePeriodValues0.add((TimePeriod) regularTimePeriod0, (Number) 1);
      assertEquals(1, timePeriodValues0.getMinMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      timePeriodValues0.add((TimePeriod) week0, (Number) 53);
      MockDate mockDate0 = new MockDate(1739, (-1952), (-4161), (-1952), 1739, 53);
      Year year0 = new Year(mockDate0, week0.DEFAULT_TIME_ZONE);
      timePeriodValues0.add((TimePeriod) year0, (Number) 1);
      assertEquals(1, timePeriodValues0.getMaxEndIndex());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      Minute minute0 = new Minute();
      timePeriodValues0.add((TimePeriod) minute0, (Number) 0);
      timePeriodValues0.add((TimePeriod) week0, (Number) 1);
      assertEquals(1, timePeriodValues0.getMinEndIndex());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      timePeriodValues0.add((TimePeriod) week0, (Number) 53);
      timePeriodValues0.delete(1739, (-205));
      assertEquals(0, timePeriodValues0.getMaxMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      // Undeclared exception!
      try { 
        timePeriodValues0.delete(53, 53);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 53, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      boolean boolean0 = timePeriodValues0.equals(timePeriodValues0);
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertTrue(boolean0);
      assertEquals("Value", timePeriodValues0.getRangeDescription());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Null 'date' argument.", "Null 'date' argument.", "Null 'date' argument.");
      Object object0 = new Object();
      boolean boolean0 = timePeriodValues0.equals(object0);
      assertEquals("Null 'date' argument.", timePeriodValues0.getRangeDescription());
      assertEquals("Null 'date' argument.", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertFalse(boolean0);
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(integer0);
      Byte byte0 = new Byte((byte)54);
      TimePeriodValues timePeriodValues1 = new TimePeriodValues(byte0, "Null 'date' argument.", "org.jfree.data.time.TimePeriodValues");
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals((-1), timePeriodValues1.getMaxMiddleIndex());
      assertFalse(boolean0);
      assertEquals((-1), timePeriodValues1.getMinStartIndex());
      assertEquals((-1), timePeriodValues1.getMinEndIndex());
      assertEquals((-1), timePeriodValues1.getMaxEndIndex());
      assertEquals((-1), timePeriodValues1.getMaxStartIndex());
      assertEquals("Null 'date' argument.", timePeriodValues1.getDomainDescription());
      assertEquals((-1), timePeriodValues1.getMinMiddleIndex());
      assertEquals("org.jfree.data.time.TimePeriodValues", timePeriodValues1.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      TimePeriodValues timePeriodValues1 = new TimePeriodValues(week0, "", "/3?1Ft");
      boolean boolean0 = timePeriodValues0.equals(timePeriodValues1);
      assertEquals((-1), timePeriodValues1.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues1.getMinEndIndex());
      assertEquals("/3?1Ft", timePeriodValues1.getRangeDescription());
      assertEquals((-1), timePeriodValues1.getMaxEndIndex());
      assertEquals((-1), timePeriodValues1.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues1.getMaxStartIndex());
      assertEquals("", timePeriodValues1.getDomainDescription());
      assertFalse(boolean0);
      assertEquals((-1), timePeriodValues1.getMinStartIndex());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Byte byte0 = new Byte((byte)5);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(byte0, "Null 'date' argument.", "Null 'date' argument.");
      TimePeriodValues timePeriodValues1 = timePeriodValues0.createCopy(52, 3245);
      timePeriodValues1.setRangeDescription("&|[^J'");
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals("&|[^J'", timePeriodValues1.getRangeDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      TimePeriodValues timePeriodValues1 = timePeriodValues0.createCopy(3245, 53);
      timePeriodValues1.add((TimePeriod) week0, (Number) 1);
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals(0, timePeriodValues1.getMinMiddleIndex());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      TimePeriodValues timePeriodValues1 = timePeriodValues0.createCopy(1, 53);
      timePeriodValues0.add((TimePeriod) week0, (Number) 53);
      timePeriodValues1.add((TimePeriod) week0, (Number) 1);
      boolean boolean0 = timePeriodValues1.equals(timePeriodValues0);
      assertEquals(1, timePeriodValues1.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Byte byte0 = new Byte((byte)5);
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(byte0, (String) null, "R[l$M!,^wMA'/X");
      timePeriodValues0.hashCode();
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals("R[l$M!,^wMA'/X", timePeriodValues0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Week week0 = new Week();
      TimePeriodValues timePeriodValues0 = new TimePeriodValues(week0);
      timePeriodValues0.hashCode();
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals("Time", timePeriodValues0.getDomainDescription());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TimePeriodValues timePeriodValues0 = new TimePeriodValues("Null 'date' argument.", "Null 'date' argument.", (String) null);
      timePeriodValues0.hashCode();
      assertEquals((-1), timePeriodValues0.getMaxMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinStartIndex());
      assertEquals((-1), timePeriodValues0.getMaxEndIndex());
      assertEquals((-1), timePeriodValues0.getMinMiddleIndex());
      assertEquals((-1), timePeriodValues0.getMinEndIndex());
      assertEquals((-1), timePeriodValues0.getMaxStartIndex());
      assertEquals("Null 'date' argument.", timePeriodValues0.getDomainDescription());
  }
}