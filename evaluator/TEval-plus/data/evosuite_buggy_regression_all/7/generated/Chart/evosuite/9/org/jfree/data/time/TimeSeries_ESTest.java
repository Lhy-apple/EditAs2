/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:36:39 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import java.util.TimeZone;
import javax.swing.JLayeredPane;
import javax.swing.table.DefaultTableCellRenderer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.text.MockSimpleDateFormat;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.jfree.data.time.Day;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Hour;
import org.jfree.data.time.Minute;
import org.jfree.data.time.Quarter;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesDataItem;
import org.jfree.data.time.Week;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TimeSeries_ESTest extends TimeSeries_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.setDomainDescription("");
      assertEquals("", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.createCopy((RegularTimePeriod) day0, (RegularTimePeriod) day0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      // Undeclared exception!
      try { 
        timeSeries0.update(2145338309, (Number) integer0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 2145338309, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, 0.0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Day day0 = new Day();
      Class<DefaultTableCellRenderer.UIResource> class0 = DefaultTableCellRenderer.UIResource.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      timeSeries0.getItems();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) quarter0, (Number) 1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Quarter, but the TimeSeries is expecting an instance of org.jfree.data.time.Day.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemCount((-707));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'maximum' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimeZone timeZone0 = TimeZone.getTimeZone("Negative 'periods' argument.");
      Second second0 = new Second(mockDate0, timeZone0);
      Class<RegularTimePeriod> class0 = RegularTimePeriod.class;
      TimeSeries timeSeries0 = new TimeSeries(second0, class0);
      timeSeries0.addOrUpdate((RegularTimePeriod) second0, (Number) 0);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.setMaximumItemCount(0);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemAge((-1946L));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'periods' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.getDataItem((RegularTimePeriod) day0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.getDataItem((RegularTimePeriod) day0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.getTimePeriods();
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.DEFAULT_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      TimeSeries timeSeries1 = new TimeSeries(integer0);
      timeSeries1.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      // Undeclared exception!
      try { 
        timeSeries0.getValue((RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.getValue((RegularTimePeriod) day0);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, 526.4876);
      timeSeries0.getValue((RegularTimePeriod) day0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.add((TimeSeriesDataItem) null, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'item' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Day day0 = new Day();
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) day0, (double) 0L);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are attempting to add an observation for the time period 14-February-2014 but the series already contains an observation for that time period. Duplicates are not permitted.  Try using the addOrUpdate() method.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.POPUP_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.add(regularTimePeriod0, (double) 2014);
      assertEquals(2, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.setMaximumItemCount(1);
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) null);
      RegularTimePeriod regularTimePeriod0 = timeSeries0.getNextTimePeriod();
      timeSeries0.add(regularTimePeriod0, (double) 1);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, (-1646.4671048), false);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      BigInteger bigInteger0 = BigInteger.ONE;
      // Undeclared exception!
      try { 
        timeSeries0.update((RegularTimePeriod) quarter0, (Number) bigInteger0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // TimeSeries.update(TimePeriod, Number):  period does not exist.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.update((RegularTimePeriod) day0, (Number) integer0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, (-491.01990297926733), true);
      TimeSeries timeSeries1 = timeSeries0.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries1.getItemCount());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertNotSame(timeSeries1, timeSeries0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.add((RegularTimePeriod) day0, (-491.01990297926733), true);
      TimeSeries timeSeries1 = new TimeSeries(day0);
      timeSeries1.addAndOrUpdate(timeSeries0);
      assertEquals(1, timeSeries1.getItemCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      // Undeclared exception!
      try { 
        timeSeries0.addOrUpdate((RegularTimePeriod) null, (Number) 4);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      MockDate mockDate0 = new MockDate();
      TimeZone timeZone0 = TimeZone.getTimeZone("Negative 'periods' argument.");
      Second second0 = new Second(mockDate0, timeZone0);
      Class<RegularTimePeriod> class0 = RegularTimePeriod.class;
      TimeSeries timeSeries0 = new TimeSeries(second0, class0);
      timeSeries0.setMaximumItemCount(0);
      timeSeries0.addOrUpdate((RegularTimePeriod) second0, (Number) 0);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Day day0 = new Day();
      Class<DefaultTableCellRenderer.UIResource> class0 = DefaultTableCellRenderer.UIResource.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      RegularTimePeriod regularTimePeriod0 = timeSeries0.getNextTimePeriod();
      TimeSeries timeSeries1 = new TimeSeries(regularTimePeriod0);
      timeSeries1.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      TimeSeries timeSeries2 = (TimeSeries)timeSeries1.clone();
      timeSeries2.add(regularTimePeriod0, (double) 0L);
      assertEquals(1, timeSeries1.getItemCount());
      
      timeSeries2.setMaximumItemAge(0L);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      TimeSeries timeSeries0 = new TimeSeries(fixedMillisecond0);
      timeSeries0.removeAgedItems((long) (-2615), false);
      timeSeries0.removeAgedItems((-2176L), false);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Day day0 = new Day();
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeriesDataItem timeSeriesDataItem0 = timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      assertNull(timeSeriesDataItem0);
      
      timeSeries0.removeAgedItems(3228L, true);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Week week0 = new Week(57, 57);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.setMaximumItemAge(37L);
      timeSeries0.addOrUpdate((RegularTimePeriod) week0, (Number) 53);
      timeSeries0.removeAgedItems((-1392409255751L), true);
      assertEquals(37L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      MockDate mockDate0 = new MockDate(9223372036854775807L);
      MockSimpleDateFormat mockSimpleDateFormat0 = new MockSimpleDateFormat();
      TimeZone timeZone0 = mockSimpleDateFormat0.getTimeZone();
      Quarter quarter0 = new Quarter(mockDate0, timeZone0);
      Class<Hour> class0 = Hour.class;
      TimeSeries timeSeries0 = new TimeSeries(quarter0, "YU:ab'", "YU:ab'", class0);
      timeSeries0.setMaximumItemAge(1L);
      timeSeries0.addOrUpdate((RegularTimePeriod) quarter0, (Number) 4);
      timeSeries0.removeAgedItems(938L, false);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Week week0 = new Week(57, 57);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      timeSeries0.clear();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      TimeSeries timeSeries0 = new TimeSeries(fixedMillisecond0);
      Integer integer0 = JLayeredPane.FRAME_CONTENT_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) fixedMillisecond0, (Number) integer0);
      timeSeries0.clear();
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      TimeSeries timeSeries0 = new TimeSeries(fixedMillisecond0);
      timeSeries0.delete((RegularTimePeriod) fixedMillisecond0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.delete((RegularTimePeriod) day0);
      assertEquals(0, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Week week0 = new Week(57, 57);
      TimeSeries timeSeries0 = new TimeSeries(week0);
      // Undeclared exception!
      try { 
        timeSeries0.delete(53, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((-2146895134), (-2146895134));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start >= 0.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy(0, (-615));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      TimeSeries timeSeries1 = timeSeries0.createCopy(1, 4);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertNotSame(timeSeries1, timeSeries0);
      assertEquals("Time", timeSeries1.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) null, (RegularTimePeriod) day0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'start' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Quarter quarter0 = new Quarter();
      TimeSeries timeSeries0 = new TimeSeries(quarter0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) quarter0, (RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'end' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      TimeSeries timeSeries0 = new TimeSeries(fixedMillisecond0);
      RegularTimePeriod regularTimePeriod0 = fixedMillisecond0.next();
      // Undeclared exception!
      try { 
        timeSeries0.createCopy(regularTimePeriod0, (RegularTimePeriod) fixedMillisecond0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start on or before end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) day0, (RegularTimePeriod) day0);
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      RegularTimePeriod regularTimePeriod0 = timeSeries0.getNextTimePeriod();
      Class<FixedMillisecond> class0 = FixedMillisecond.class;
      TimeSeries timeSeries1 = new TimeSeries(regularTimePeriod0, class0);
      TimeSeriesDataItem timeSeriesDataItem0 = timeSeries1.addOrUpdate(regularTimePeriod0, (Number) integer0);
      assertNull(timeSeriesDataItem0);
      
      TimeSeries timeSeries2 = timeSeries1.createCopy((RegularTimePeriod) day0, (RegularTimePeriod) day0);
      assertEquals(Integer.MAX_VALUE, timeSeries2.getMaximumItemCount());
      assertEquals("Time", timeSeries2.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries2.getMaximumItemAge());
      assertEquals("Value", timeSeries2.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      TimeSeries timeSeries0 = new TimeSeries(fixedMillisecond0);
      boolean boolean0 = timeSeries0.equals(timeSeries0);
      assertTrue(boolean0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Day day0 = new Day();
      Class<DefaultTableCellRenderer.UIResource> class0 = DefaultTableCellRenderer.UIResource.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      boolean boolean0 = timeSeries0.equals(", but the TimeSeries is expecting an instance of ");
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertFalse(boolean0);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      assertTrue(timeSeries1.equals((Object)timeSeries0));
      
      timeSeries1.setDescription("*^%1`");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertFalse(timeSeries1.equals((Object)timeSeries0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Class<Minute> class0 = Minute.class;
      TimeSeries timeSeries1 = new TimeSeries(day0, "0}Dc}Dinu{Us2", "0}Dc}Dinu{Us2", class0);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("0}Dc}Dinu{Us2", timeSeries1.getDomainDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("0}Dc}Dinu{Us2", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.setRangeDescription("S9gmdZ");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals("S9gmdZ", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.setMaximumItemAge(659L);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(659L, timeSeries1.getMaximumItemAge());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      timeSeries1.setMaximumItemCount(2000);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(2000, timeSeries1.getMaximumItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Day day0 = new Day();
      Class<DefaultTableCellRenderer.UIResource> class0 = DefaultTableCellRenderer.UIResource.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      TimeSeries timeSeries1 = new TimeSeries(day0);
      assertTrue(timeSeries1.equals((Object)timeSeries0));
      
      timeSeries1.add((RegularTimePeriod) day0, (double) 0L);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertFalse(timeSeries1.equals((Object)timeSeries0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Day day0 = new Day();
      Class<Hour> class0 = Hour.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      TimeSeriesDataItem timeSeriesDataItem0 = timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      assertNull(timeSeriesDataItem0);
      
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertTrue(boolean0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertNotSame(timeSeries1, timeSeries0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Day day0 = new Day();
      Class<Hour> class0 = Hour.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      timeSeries0.setMaximumItemAge(0L);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      Object object0 = timeSeries0.clone();
      RegularTimePeriod regularTimePeriod0 = timeSeries0.getNextTimePeriod();
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) integer0);
      boolean boolean0 = timeSeries0.equals(object0);
      assertEquals(1, timeSeries0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Day day0 = new Day();
      Class<FixedMillisecond> class0 = FixedMillisecond.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, (String) null, (String) null, class0);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0, (Class) null);
      timeSeries0.hashCode();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Day day0 = new Day();
      Class<DefaultTableCellRenderer.UIResource> class0 = DefaultTableCellRenderer.UIResource.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, class0);
      Integer integer0 = JLayeredPane.DRAG_LAYER;
      TimeSeriesDataItem timeSeriesDataItem0 = timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      assertNull(timeSeriesDataItem0);
      
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      RegularTimePeriod regularTimePeriod0 = timeSeries0.getNextTimePeriod();
      timeSeries0.add(regularTimePeriod0, (double) 2014);
      timeSeries0.hashCode();
      assertEquals(2, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Day day0 = new Day();
      Integer integer0 = JLayeredPane.MODAL_LAYER;
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      RegularTimePeriod regularTimePeriod0 = timeSeries0.getNextTimePeriod();
      timeSeries0.addOrUpdate(regularTimePeriod0, (Number) integer0);
      RegularTimePeriod regularTimePeriod1 = timeSeries0.getNextTimePeriod();
      timeSeries0.addOrUpdate(regularTimePeriod1, (Number) integer0);
      timeSeries0.hashCode();
      assertEquals(3, timeSeries0.getItemCount());
  }
}