/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:20:28 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.PushbackInputStream;
import org.apache.commons.codec.binary.Base64InputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileInputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64InputStream_ESTest extends Base64InputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DataInputStream dataInputStream0 = new DataInputStream((InputStream) null);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(dataInputStream0);
      byte[] byteArray0 = new byte[0];
      Base64InputStream base64InputStream0 = new Base64InputStream(pushbackInputStream0, false, (-4613), byteArray0);
      assertFalse(base64InputStream0.markSupported());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Base64InputStream base64InputStream0 = new Base64InputStream((InputStream) null);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 8192, 8192);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      boolean boolean0 = base64InputStream0.markSupported();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, false);
      int int0 = base64InputStream0.read();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[1] = (byte)100;
      byteArray0[3] = (byte)97;
      byteArray0[4] = (byte)55;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, false);
      base64InputStream0.read();
      int int0 = base64InputStream0.read();
      assertEquals(174, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = MockFile.createTempFile("u?rW31=!", "u?rW31=!");
      MockFileInputStream mockFileInputStream0 = new MockFileInputStream(file0);
      Base64InputStream base64InputStream0 = new Base64InputStream(mockFileInputStream0, false);
      // Undeclared exception!
      try { 
        base64InputStream0.read((byte[]) null, (-2886), (-1));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Base64InputStream base64InputStream0 = new Base64InputStream((InputStream) null, true);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, (-3257), (-3257));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 65, (-1708));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base64InputStream base64InputStream0 = new Base64InputStream((InputStream) null);
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 0, 2061);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      byte[] byteArray1 = new byte[0];
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read(byteArray1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read();
      assertEquals(65, int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read(byteArray0, 1, 1);
      assertArrayEquals(new byte[] {(byte)0, (byte)65, (byte)0}, byteArray0);
      assertEquals(1, int0);
  }
}