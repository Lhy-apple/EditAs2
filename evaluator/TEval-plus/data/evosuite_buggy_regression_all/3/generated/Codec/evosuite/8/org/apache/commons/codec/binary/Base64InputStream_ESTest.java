/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:27:14 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import org.apache.commons.codec.binary.Base64InputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64InputStream_ESTest extends Base64InputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      File file0 = MockFile.createTempFile("#4U5\"_akzk>FR\"b=$6", "#4U5\"_akzk>FR\"b=$6", (File) null);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      Base64InputStream base64InputStream0 = new Base64InputStream(mockFileInputStream0, true);
      boolean boolean0 = base64InputStream0.markSupported();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      int int0 = base64InputStream0.read();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      File file0 = MockFile.createTempFile("#4U5\"_akzk>FR\"b=$6", "#4U5\"_akzk>FR\"b=$6", (File) null);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      Base64InputStream base64InputStream0 = new Base64InputStream(mockFileInputStream0, true);
      // Undeclared exception!
      try { 
        base64InputStream0.read((byte[]) null, 2577, 1821);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, (-1), (-1));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 2380, (-1308));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 4095, 20);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      File file0 = MockFile.createTempFile("#4U5\"_akzk>FR\"b=$6", "#4U5\"_akzk>FR\"b=$6", (File) null);
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      Base64InputStream base64InputStream0 = new Base64InputStream(mockFileInputStream0, true);
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 0, (int) (byte)47);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true, (-11), byteArray0);
      int int0 = base64InputStream0.read(byteArray0, 0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true, 2, byteArray0);
      base64InputStream0.read();
      int int0 = base64InputStream0.read();
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(65, int0);
  }
}