/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:56:20 GMT 2023
 */

package com.google.gson;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.DefaultDateTypeAdapter;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import java.sql.Date;
import java.sql.Timestamp;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockDate;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultDateTypeAdapter_ESTest extends DefaultDateTypeAdapter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<Date> class0 = Date.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, 0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0);
      String string0 = defaultDateTypeAdapter0.toString();
      assertEquals("DefaultDateTypeAdapter(SimpleDateFormat)", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, "");
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Class<Date> class0 = Date.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0);
      Date date0 = new Date(1338);
      JsonElement jsonElement0 = defaultDateTypeAdapter0.toJsonTree(date0);
      java.util.Date date1 = defaultDateTypeAdapter0.fromJsonTree(jsonElement0);
      assertFalse(date1.equals((Object)date0));
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, 0, 0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(0, 0);
      JsonArray jsonArray0 = new JsonArray();
      // Undeclared exception!
      try { 
        defaultDateTypeAdapter0.fromJsonTree(jsonArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // The date should be a string value
         //
         verifyException("com.google.gson.DefaultDateTypeAdapter", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = null;
      try {
        defaultDateTypeAdapter0 = new DefaultDateTypeAdapter((Class<? extends java.util.Date>) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Date type must be one of class java.util.Date, class java.sql.Timestamp, or class java.sql.Date but was null
         //
         verifyException("com.google.gson.DefaultDateTypeAdapter", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(2, 2);
      JsonElement jsonElement0 = defaultDateTypeAdapter0.toJsonTree((java.util.Date) null);
      assertFalse(jsonElement0.isJsonArray());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(0, 0);
      MockDate mockDate0 = new MockDate((-130), 0, (-130), 0, (-2), (-2));
      JsonElement jsonElement0 = defaultDateTypeAdapter0.toJsonTree(mockDate0);
      java.util.Date date0 = defaultDateTypeAdapter0.fromJsonTree(jsonElement0);
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      MockDate mockDate0 = new MockDate(0, 0, (-2), 0, 0, 1338);
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0);
      JsonElement jsonElement0 = defaultDateTypeAdapter0.toJsonTree(mockDate0);
      java.util.Date date0 = defaultDateTypeAdapter0.fromJsonTree(jsonElement0);
      assertEquals("2014-02-14 20:21:21.32", date0.toString());
  }
}