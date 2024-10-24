/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:23:42 GMT 2023
 */

package org.apache.commons.lang3;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.Locale;
import org.apache.commons.lang3.SerializationUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SerializationUtils_ESTest extends SerializationUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      Locale locale1 = SerializationUtils.clone(locale0);
      assertEquals("fra", locale1.getISO3Language());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SerializationUtils serializationUtils0 = new SerializationUtils();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String string0 = SerializationUtils.clone((String) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String[] stringArray0 = Locale.getISOCountries();
      // Undeclared exception!
      try { 
        SerializationUtils.serialize((Serializable) stringArray0, (OutputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The OutputStream must not be null
         //
         verifyException("org.apache.commons.lang3.SerializationUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      DataOutputStream dataOutputStream0 = new DataOutputStream((OutputStream) null);
      // Undeclared exception!
      try { 
        SerializationUtils.serialize((Serializable) locale0, (OutputStream) dataOutputStream0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.DataOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Integer integer0 = new Integer(2969);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("9Q[[%<s4=zPK&;s%`=o");
      // Undeclared exception!
      try { 
        SerializationUtils.serialize((Serializable) integer0, (OutputStream) mockFileOutputStream0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // org.evosuite.runtime.mock.java.lang.MockThrowable: Error in writing to file
         //
         verifyException("org.apache.commons.lang3.SerializationUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = SerializationUtils.serialize((Serializable) (short)5);
      Object object0 = SerializationUtils.deserialize(byteArray0);
      assertEquals((short)5, object0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        SerializationUtils.deserialize((InputStream) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The InputStream must not be null
         //
         verifyException("org.apache.commons.lang3.SerializationUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        SerializationUtils.deserialize(byteArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.io.EOFException
         //
         verifyException("org.apache.commons.lang3.SerializationUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = SerializationUtils.serialize((Serializable) (short)5);
      byteArray0[13] = (byte)1;
      // Undeclared exception!
      try { 
        SerializationUtils.deserialize(byteArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.ClassNotFoundException: java.\u0001ang.Short
         //
         verifyException("org.apache.commons.lang3.SerializationUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        SerializationUtils.deserialize((byte[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The byte[] must not be null
         //
         verifyException("org.apache.commons.lang3.SerializationUtils", e);
      }
  }
}
