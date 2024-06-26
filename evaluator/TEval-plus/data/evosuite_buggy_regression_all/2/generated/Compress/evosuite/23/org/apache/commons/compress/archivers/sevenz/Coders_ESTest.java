/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:46:48 GMT 2023
 */

package org.apache.commons.compress.archivers.sevenz;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.archivers.sevenz.Coder;
import org.apache.commons.compress.archivers.sevenz.Coders;
import org.apache.commons.compress.archivers.sevenz.SevenZMethod;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Coders_ESTest extends Coders_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Coders coders0 = new Coders();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Coder coder0 = new Coder();
      byte[] byteArray0 = new byte[8];
      coder0.properties = byteArray0;
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode((InputStream) null, coder0, coder0.properties);
      Coders.LZMADecoder coders_LZMADecoder0 = new Coders.LZMADecoder();
      // Undeclared exception!
      try { 
        coders_LZMADecoder0.decode(inputStream0, coder0, byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("javax.crypto.CipherInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      Coder coder0 = new Coder();
      try { 
        coders_BZIP2Decoder0.decode((InputStream) null, coder0, (byte[]) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No InputStream
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      SevenZMethod sevenZMethod0 = SevenZMethod.BZIP2;
      byte[] byteArray0 = new byte[14];
      try { 
        Coders.addEncoder(pipedOutputStream0, sevenZMethod0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      SevenZMethod sevenZMethod0 = SevenZMethod.DEFLATE;
      byte[] byteArray0 = new byte[1];
      OutputStream outputStream0 = Coders.addEncoder(pipedOutputStream0, sevenZMethod0, byteArray0);
      assertNotNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      SevenZMethod sevenZMethod0 = SevenZMethod.COPY;
      OutputStream outputStream0 = Coders.addEncoder(pipedOutputStream0, sevenZMethod0, (byte[]) null);
      assertSame(pipedOutputStream0, outputStream0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Coder coder0 = new Coder();
      byte[] byteArray0 = new byte[1];
      coder0.decompressionMethodId = byteArray0;
      coder0.properties = byteArray0;
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(coder0.properties);
      InputStream inputStream0 = Coders.addDecoder(byteArrayInputStream0, coder0, byteArray0);
      assertSame(inputStream0, byteArrayInputStream0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SevenZMethod sevenZMethod0 = SevenZMethod.BZIP2;
      Coders.CopyDecoder coders_CopyDecoder0 = new Coders.CopyDecoder();
      Coders.CoderId coders_CoderId0 = new Coders.CoderId(sevenZMethod0, coders_CopyDecoder0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      SevenZMethod sevenZMethod0 = SevenZMethod.AES256SHA256;
      // Undeclared exception!
      try { 
        Coders.addEncoder((OutputStream) null, sevenZMethod0, byteArray0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // method doesn't support writing
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$CoderBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Coder coder0 = new Coder();
      byte[] byteArray0 = new byte[5];
      try { 
        Coders.addDecoder((InputStream) null, coder0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unsupported compression method null
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      File file0 = MockFile.createTempFile("@F}~W!$zBR44}>p", (String) null);
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      try { 
        Coders.addEncoder(mockPrintStream0, (SevenZMethod) null, (byte[]) null);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unsupported compression method null
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      byte[] byteArray0 = new byte[9];
      coder0.properties = byteArray0;
      byteArray0[1] = (byte)101;
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode((InputStream) null, coder0, byteArray0);
      ObjectInputStream objectInputStream0 = null;
      try {
        objectInputStream0 = new ObjectInputStream(inputStream0);
        fail("Expecting exception: IOException");
      
      } catch(Throwable e) {
         //
         // Salt size + IV size too long
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$AES256SHA256Decoder$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Coder coder0 = new Coder();
      byte[] byteArray0 = new byte[9];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      byteArrayInputStream0.skip((byte)7);
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      InputStream inputStream0 = coders_DeflateDecoder0.decode(byteArrayInputStream0, coder0, byteArray0);
      ObjectInputStream objectInputStream0 = null;
      try {
        objectInputStream0 = new ObjectInputStream(inputStream0);
        fail("Expecting exception: EOFException");
      
      } catch(Throwable e) {
         //
         // Unexpected end of ZLIB input stream
         //
         verifyException("java.util.zip.InflaterInputStream", e);
      }
  }
}
