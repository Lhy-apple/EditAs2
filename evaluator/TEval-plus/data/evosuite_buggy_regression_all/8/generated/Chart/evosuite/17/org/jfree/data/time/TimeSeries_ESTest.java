/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:54:20 GMT 2023
 */

package org.jfree.data.time;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import javax.swing.JLayeredPane;
import javax.swing.JPopupMenu;
import javax.swing.undo.UndoManager;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.data.time.Day;
import org.jfree.data.time.FixedMillisecond;
import org.jfree.data.time.Hour;
import org.jfree.data.time.Minute;
import org.jfree.data.time.RegularTimePeriod;
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
      BigInteger bigInteger0 = BigInteger.ONE;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      Object object0 = timeSeries0.clone();
      boolean boolean0 = timeSeries0.equals(object0);
      assertEquals(1, timeSeries0.getItemCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.setDomainDescription("");
      assertEquals("", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.update(59, (Number) 0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 59, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.getItems();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.getNextTimePeriod();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (Number) 0);
      timeSeries0.getValue((RegularTimePeriod) minute0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) minute0, (double) 0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are trying to add data where the time period class is org.jfree.data.time.Minute, but the TimeSeries is expecting an instance of org.jfree.data.time.Day.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = new TimeSeries(day0);
      timeSeries1.addOrUpdate((RegularTimePeriod) day0, 0.0);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(1, timeSeries1.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.setMaximumItemCount(0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 59);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemCount((-1762));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'maximum' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 59);
      timeSeries0.setMaximumItemCount(0);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.setMaximumItemAge((-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Negative 'periods' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.getDataItem((RegularTimePeriod) minute0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.POPUP_LAYER;
      timeSeries0.add((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.getDataItem((RegularTimePeriod) day0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 0);
      timeSeries0.getTimePeriods();
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      timeSeries0.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      TimeSeries timeSeries1 = new TimeSeries(day0);
      timeSeries1.getTimePeriodsUniqueToOtherSeries(timeSeries0);
      assertEquals(1, timeSeries0.getItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TimeSeries timeSeries0 = new TimeSeries("");
      // Undeclared exception!
      try { 
        timeSeries0.getDataItem((RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.getValue((RegularTimePeriod) minute0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
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
  public void test20()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      // Undeclared exception!
      try { 
        timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // You are attempting to add an observation for the time period 14-February-2014 but the series already contains an observation for that time period. Duplicates are not permitted.  Try using the addOrUpdate() method.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.add(regularTimePeriod0, (Number) bigInteger0);
      timeSeries0.createCopy(regularTimePeriod0, (RegularTimePeriod) day0);
      assertEquals(2, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      TimeSeries timeSeries0 = new TimeSeries("");
      Day day0 = new Day();
      timeSeries0.setMaximumItemCount(0);
      Integer integer0 = new Integer(0);
      TimeSeriesDataItem timeSeriesDataItem0 = new TimeSeriesDataItem((RegularTimePeriod) day0, (Number) integer0);
      timeSeries0.add(timeSeriesDataItem0, false);
      assertEquals(0, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.update((RegularTimePeriod) minute0, (Number) 0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // TimeSeries.update(TimePeriod, Number):  period does not exist.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.update((RegularTimePeriod) day0, (Number) bigInteger0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      TimeSeries timeSeries1 = timeSeries0.addAndOrUpdate(timeSeries0);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertNotSame(timeSeries1, timeSeries0);
      assertEquals(1, timeSeries1.getItemCount());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Time", timeSeries1.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond();
      Class<TimeSeries> class0 = TimeSeries.class;
      TimeSeries timeSeries0 = new TimeSeries(fixedMillisecond0, class0);
      Class<JPopupMenu> class1 = JPopupMenu.class;
      TimeSeries timeSeries1 = new TimeSeries(fixedMillisecond0, class1);
      timeSeries1.addOrUpdate((RegularTimePeriod) fixedMillisecond0, 2782.419175224);
      TimeSeries timeSeries2 = timeSeries0.addAndOrUpdate(timeSeries1);
      assertEquals(1, timeSeries1.getItemCount());
      assertFalse(timeSeries2.equals((Object)timeSeries0));
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.addOrUpdate((RegularTimePeriod) null, (double) 59);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'period' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.add(regularTimePeriod0, (Number) bigInteger0);
      timeSeries0.setMaximumItemAge(0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.setMaximumItemAge(0);
      timeSeries0.add(regularTimePeriod0, (Number) bigInteger0);
      assertEquals(1, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.removeAgedItems((long) 0, true);
      timeSeries0.removeAgedItems((long) 59, true);
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.ONE;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      timeSeries0.removeAgedItems((long) 0, false);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.ZERO;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      
      timeSeries0.setMaximumItemAge(0);
      timeSeries0.removeAgedItems(53477366400000L, true);
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.ZERO;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      
      timeSeries0.setMaximumItemAge(0);
      timeSeries0.removeAgedItems(53477366400000L, false);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.clear();
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.addOrUpdate((RegularTimePeriod) minute0, (double) 59);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.clear();
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      timeSeries0.delete((RegularTimePeriod) minute0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      timeSeries0.addOrUpdate((RegularTimePeriod) day0, (Number) integer0);
      assertEquals(1, timeSeries0.getItemCount());
      
      timeSeries0.delete((RegularTimePeriod) day0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.delete(3293, 7);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
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
        timeSeries0.createCopy((-4159), (-696));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start >= 0.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      // Undeclared exception!
      try { 
        timeSeries0.clone();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start <= end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      TimeSeries timeSeries1 = timeSeries0.createCopy(0, 0);
      assertNotSame(timeSeries1, timeSeries0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) null, (RegularTimePeriod) minute0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'start' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) minute0, (RegularTimePeriod) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'end' argument.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      // Undeclared exception!
      try { 
        timeSeries0.createCopy((RegularTimePeriod) day0, regularTimePeriod0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires start on or before end.
         //
         verifyException("org.jfree.data.time.TimeSeries", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) day0, (RegularTimePeriod) day0);
      timeSeries1.setRangeDescription("Requires start <= end.");
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals("Requires start <= end.", timeSeries1.getRangeDescription());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      TimeSeries timeSeries1 = timeSeries0.createCopy(regularTimePeriod0, regularTimePeriod0);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals(0, timeSeries1.getItemCount());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      boolean boolean0 = timeSeries0.equals(timeSeries0);
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertTrue(boolean0);
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.ONE;
      boolean boolean0 = timeSeries0.equals(bigInteger0);
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertFalse(boolean0);
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Minute minute0 = new Minute();
      TimeSeries timeSeries0 = new TimeSeries(minute0);
      Week week0 = new Week();
      TimeSeries timeSeries1 = new TimeSeries(week0);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Time", timeSeries1.getDomainDescription());
      assertEquals("Value", timeSeries1.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      Class<UndoManager> class0 = UndoManager.class;
      TimeSeries timeSeries1 = new TimeSeries(day0, (String) null, (String) null, class0);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(Integer.MAX_VALUE, timeSeries1.getMaximumItemCount());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertFalse(boolean0);
      assertEquals(9223372036854775807L, timeSeries1.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.setMaximumItemAge(5788L);
      TimeSeries timeSeries1 = new TimeSeries(day0);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(5788L, timeSeries0.getMaximumItemAge());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      TimeSeries timeSeries1 = timeSeries0.createCopy((RegularTimePeriod) day0, (RegularTimePeriod) day0);
      timeSeries1.setMaximumItemCount(4105);
      boolean boolean0 = timeSeries1.equals(timeSeries0);
      assertEquals(4105, timeSeries1.getMaximumItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.ONE;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      TimeSeries timeSeries1 = (TimeSeries)timeSeries0.clone();
      FixedMillisecond fixedMillisecond0 = new FixedMillisecond(2014);
      timeSeries1.addOrUpdate((RegularTimePeriod) fixedMillisecond0, 0.0);
      boolean boolean0 = timeSeries0.equals(timeSeries1);
      assertEquals(1, timeSeries0.getItemCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Day day0 = new Day();
      Class<Hour> class0 = Hour.class;
      TimeSeries timeSeries0 = new TimeSeries(day0, (String) null, (String) null, class0);
      timeSeries0.hashCode();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      timeSeries0.hashCode();
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals("Time", timeSeries0.getDomainDescription());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.ONE;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      timeSeries0.hashCode();
      assertEquals(9223372036854775807L, timeSeries0.getMaximumItemAge());
      assertEquals("Time", timeSeries0.getDomainDescription());
      assertEquals("Value", timeSeries0.getRangeDescription());
      assertEquals(Integer.MAX_VALUE, timeSeries0.getMaximumItemCount());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.add(regularTimePeriod0, (Number) bigInteger0);
      timeSeries0.hashCode();
      assertEquals(2, timeSeries0.getItemCount());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Day day0 = new Day();
      TimeSeries timeSeries0 = new TimeSeries(day0);
      BigInteger bigInteger0 = BigInteger.TEN;
      timeSeries0.add((RegularTimePeriod) day0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod0 = day0.previous();
      timeSeries0.add(regularTimePeriod0, (Number) bigInteger0);
      RegularTimePeriod regularTimePeriod1 = regularTimePeriod0.previous();
      timeSeries0.add(regularTimePeriod1, (Number) bigInteger0);
      timeSeries0.hashCode();
      assertEquals(3, timeSeries0.getItemCount());
  }
}
