/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:36:34 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PushbackInputStream;
import java.io.StreamCorruptedException;
import org.apache.commons.codec.binary.Base64InputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64InputStream_ESTest extends Base64InputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0, 480);
      byte[] byteArray0 = new byte[1];
      Base64InputStream base64InputStream0 = new Base64InputStream(pipedInputStream0, false, 480, byteArray0);
      assertFalse(base64InputStream0.markSupported());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      int int0 = base64InputStream0.read();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream();
      BufferedInputStream bufferedInputStream0 = new BufferedInputStream(pipedInputStream0, 1);
      Base64InputStream base64InputStream0 = new Base64InputStream(bufferedInputStream0);
      boolean boolean0 = base64InputStream0.markSupported();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)113;
      byteArray0[5] = (byte)84;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0);
      int int0 = base64InputStream0.read();
      assertEquals(169, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read();
      assertEquals(65, int0);
      
      int int1 = base64InputStream0.read();
      assertEquals(65, int1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      PipedInputStream pipedInputStream0 = new PipedInputStream(pipedOutputStream0);
      Base64InputStream base64InputStream0 = new Base64InputStream(pipedInputStream0, true);
      // Undeclared exception!
      try { 
        base64InputStream0.read((byte[]) null, 64, 64);
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
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 1709, (-487));
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
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)6, (-49));
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      Base64InputStream base64InputStream0 = new Base64InputStream(pushbackInputStream0);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, (int) (byte)6, 6);
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
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 1, 1);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read(byteArray0, 1, 0);
      assertEquals(0, int0);
      assertEquals(1, byteArrayInputStream0.available());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      ObjectInputStream objectInputStream0 = null;
      try {
        objectInputStream0 = new ObjectInputStream(base64InputStream0);
        fail("Expecting exception: StreamCorruptedException");
      
      } catch(Throwable e) {
         //
         // invalid stream header: 41413D3D
         //
         verifyException("java.io.ObjectInputStream", e);
      }
  }
}
