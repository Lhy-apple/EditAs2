/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:03:07 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.SequenceInputStream;
import org.apache.commons.codec.binary.Base64InputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64InputStream_ESTest extends Base64InputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, false, 1647, byteArray0);
      boolean boolean0 = base64InputStream0.markSupported();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, false);
      int int0 = base64InputStream0.read();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[23];
      byteArray0[0] = (byte)79;
      byteArray0[1] = (byte)79;
      byteArray0[2] = (byte)79;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(byteArrayInputStream0, byteArrayInputStream0);
      Base64InputStream base64InputStream0 = new Base64InputStream(sequenceInputStream0, false);
      int int0 = base64InputStream0.read();
      assertEquals(56, int0);
      
      int int1 = base64InputStream0.read();
      assertEquals(227, int1);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Base64InputStream base64InputStream0 = new Base64InputStream((InputStream) null);
      // Undeclared exception!
      try { 
        base64InputStream0.read((byte[]) null, (-1946), (-1946));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, false);
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
  public void test5()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, 65, (-916));
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
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(byteArrayInputStream0, byteArrayInputStream0);
      Base64InputStream base64InputStream0 = new Base64InputStream(sequenceInputStream0, false);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, (int) (byte)79, (int) (byte)79);
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
      byte[] byteArray0 = new byte[2];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, false, 1647, byteArray0);
      // Undeclared exception!
      try { 
        base64InputStream0.read(byteArray0, (int) (byte)0, 5327);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64InputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read(byteArray0, 0, 0);
      assertEquals(12, byteArrayInputStream0.available());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Base64InputStream base64InputStream0 = new Base64InputStream(byteArrayInputStream0, true);
      int int0 = base64InputStream0.read(byteArray0, 1, 1);
      assertArrayEquals(new byte[] {(byte)0, (byte)65, (byte)0, (byte)0}, byteArray0);
      assertEquals(1, int0);
  }
}