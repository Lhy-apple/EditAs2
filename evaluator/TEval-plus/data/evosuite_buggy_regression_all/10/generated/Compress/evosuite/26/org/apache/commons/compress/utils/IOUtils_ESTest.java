/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:08:50 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.InputStream;
import java.io.PipedInputStream;
import java.io.PushbackInputStream;
import org.apache.commons.compress.utils.IOUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class IOUtils_ESTest extends IOUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)74);
      byte[] byteArray1 = IOUtils.toByteArray(byteArrayInputStream0);
      assertEquals(6, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)74);
      int int0 = IOUtils.readFully((InputStream) byteArrayInputStream0, byteArray0);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(6, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)74);
      long long0 = IOUtils.skip(byteArrayInputStream0, 0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)74);
      long long0 = IOUtils.skip(byteArrayInputStream0, 329L);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(7L, long0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      // Undeclared exception!
      try { 
        IOUtils.readFully((InputStream) null, (byte[]) null, (-1056), (-1056));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.IOUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 0, 3513);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      // Undeclared exception!
      try { 
        IOUtils.readFully((InputStream) pushbackInputStream0, byteArray0, (-536), 0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.IOUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      PipedInputStream pipedInputStream0 = new PipedInputStream(476);
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        IOUtils.readFully((InputStream) pipedInputStream0, byteArray0, 5087, 1231);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.IOUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)74);
      DataInputStream dataInputStream0 = new DataInputStream(byteArrayInputStream0);
      dataInputStream0.readUnsignedByte();
      int int0 = IOUtils.readFully((InputStream) byteArrayInputStream0, byteArray0);
      assertEquals(0, byteArrayInputStream0.available());
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      IOUtils.closeQuietly((Closeable) null);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, (byte)0, (byte)74);
      DataInputStream dataInputStream0 = new DataInputStream(byteArrayInputStream0);
      IOUtils.closeQuietly(dataInputStream0);
  }
}
