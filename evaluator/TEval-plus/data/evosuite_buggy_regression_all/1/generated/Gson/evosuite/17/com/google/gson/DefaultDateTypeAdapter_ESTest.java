/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:29:46 GMT 2023
 */

package com.google.gson;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.gson.DefaultDateTypeAdapter;
import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;
import com.google.gson.stream.JsonWriter;
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
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, 0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, 0, 1);
      String string0 = defaultDateTypeAdapter0.toString();
      assertEquals("DefaultDateTypeAdapter(SimpleDateFormat)", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Class<Date> class0 = Date.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = null;
      try {
        defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, "PB{");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Illegal pattern character 'P'
         //
         verifyException("java.text.SimpleDateFormat", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0, 0, 1);
      MockDate mockDate0 = new MockDate(0, 0, 0);
      JsonElement jsonElement0 = defaultDateTypeAdapter0.toJsonTree(mockDate0);
      Timestamp timestamp0 = (Timestamp)defaultDateTypeAdapter0.fromJsonTree(jsonElement0);
      assertEquals(320000000, timestamp0.getNanos());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Class<Timestamp> class0 = Timestamp.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0);
      Integer integer0 = new Integer(0);
      JsonPrimitive jsonPrimitive0 = new JsonPrimitive(integer0);
      // Undeclared exception!
      try { 
        defaultDateTypeAdapter0.fromJsonTree(jsonPrimitive0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // The date should be a string value
         //
         verifyException("com.google.gson.DefaultDateTypeAdapter", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(0, 0);
      // Undeclared exception!
      try { 
        defaultDateTypeAdapter0.write((JsonWriter) null, (java.util.Date) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.gson.DefaultDateTypeAdapter", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Class<Date> class0 = Date.class;
      DefaultDateTypeAdapter defaultDateTypeAdapter0 = new DefaultDateTypeAdapter(class0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
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
}