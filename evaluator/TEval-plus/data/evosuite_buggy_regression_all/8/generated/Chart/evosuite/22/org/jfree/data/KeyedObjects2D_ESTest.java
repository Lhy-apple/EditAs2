/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:54:39 GMT 2023
 */

package org.jfree.data;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.jfree.data.KeyedObjects;
import org.jfree.data.KeyedObjects2D;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class KeyedObjects2D_ESTest extends KeyedObjects2D_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.getRowKey(0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.removeColumn((Comparable) "org.jfree.data.UnknownKeyException");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Column key (org.jfree.data.UnknownKeyException) not recognised.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.removeColumn(653);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 653, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.removeRow((Comparable) "RbyS7&IW6");
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.hashCode();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.addObject("Null 'columnKey' argument.", "Null 'columnKey' argument.", "Null 'columnKey' argument.");
      keyedObjects2D0.setObject("org.jfree.data.UnknownKeyException", "org.jfree.data.UnknownKeyException", "org.jfree.data.UnknownKeyException");
      Object object0 = keyedObjects2D0.clone();
      boolean boolean0 = keyedObjects2D0.equals(object0);
      assertEquals(2, keyedObjects2D0.getColumnCount());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.getObject((Comparable) "org.jfree.data.UnknownKeyException", (Comparable) "org.jfree.data.UnknownKeyException");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Row key (org.jfree.data.UnknownKeyException) not recognised.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.getObject((Comparable) null, (Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'rowKey' argument.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.getObject((Comparable) "q", (Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'columnKey' argument.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.setObject("org.jfree.data.UnknownKeyException", "org.jfree.data.UnknownKeyException", "org.jfree.data.UnknownKeyException");
      Object object0 = keyedObjects2D0.getObject((Comparable) "org.jfree.data.UnknownKeyException", (Comparable) "org.jfree.data.UnknownKeyException");
      assertEquals("org.jfree.data.UnknownKeyException", object0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.setObject("'ZeY5mytTjock_", "'ZeY5mytTjock_", "'ZeY5mytTjock_");
      // Undeclared exception!
      try { 
        keyedObjects2D0.getObject((Comparable) "'ZeY5mytTjock_", (Comparable) 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Column key (1) not recognised.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.removeObject((Comparable) null, (Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'rowKey' argument.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      // Undeclared exception!
      try { 
        keyedObjects2D0.setObject("_'+$`z!Fs496u=kD,6", "_'+$`z!Fs496u=kD,6", (Comparable) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'columnKey' argument.
         //
         verifyException("org.jfree.data.KeyedObjects2D", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      Comparable<String> comparable0 = (Comparable<String>) mock(Comparable.class, new ViolatedAssumptionAnswer());
      keyedObjects2D0.removeObject(comparable0, 0);
      keyedObjects2D0.setObject("", "", "");
      KeyedObjects2D keyedObjects2D1 = (KeyedObjects2D)keyedObjects2D0.clone();
      Integer integer0 = new Integer(0);
      keyedObjects2D1.setObject("", "", integer0);
      boolean boolean0 = keyedObjects2D0.equals(keyedObjects2D1);
      assertEquals(1, keyedObjects2D0.getRowCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.addObject("", "", "");
      Integer integer0 = Integer.valueOf((-1));
      keyedObjects2D0.removeObject("", integer0);
      assertEquals(1, keyedObjects2D0.getRowCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      boolean boolean0 = keyedObjects2D0.equals(keyedObjects2D0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      KeyedObjects keyedObjects0 = new KeyedObjects();
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      boolean boolean0 = keyedObjects2D0.equals(keyedObjects0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.setObject("+v8bqp+)PA1&#K", "+v8bqp+)PA1&#K", "+v8bqp+)PA1&#K");
      KeyedObjects2D keyedObjects2D1 = new KeyedObjects2D();
      boolean boolean0 = keyedObjects2D0.equals(keyedObjects2D1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.setObject("org.jfree.data.UnknownKeyException", "org.jfree.data.UnknownKeyException", "org.jfree.data.UnknownKeyException");
      KeyedObjects2D keyedObjects2D1 = (KeyedObjects2D)keyedObjects2D0.clone();
      keyedObjects2D1.removeColumn((Comparable) "org.jfree.data.UnknownKeyException");
      boolean boolean0 = keyedObjects2D0.equals(keyedObjects2D1);
      assertEquals(1, keyedObjects2D0.getColumnCount());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      KeyedObjects2D keyedObjects2D0 = new KeyedObjects2D();
      keyedObjects2D0.setObject("+_8bqp+PA1&#K", "+_8bqp+PA1&#K", "+_8bqp+PA1&#K");
      KeyedObjects2D keyedObjects2D1 = (KeyedObjects2D)keyedObjects2D0.clone();
      assertTrue(keyedObjects2D1.equals((Object)keyedObjects2D0));
      
      keyedObjects2D1.addObject(keyedObjects2D0, "+_8bqp+PA1&#K", "+_8bqp+PA1&#K");
      boolean boolean0 = keyedObjects2D0.equals(keyedObjects2D1);
      assertFalse(keyedObjects2D1.equals((Object)keyedObjects2D0));
      assertFalse(boolean0);
  }
}
