/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:28:49 GMT 2023
 */

package org.apache.commons.compress.archivers.sevenz;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.sevenz.Coder;
import org.apache.commons.compress.archivers.sevenz.Coders;
import org.apache.commons.compress.archivers.sevenz.SevenZMethod;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Coders_ESTest extends Coders_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Coders coders0 = new Coders();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      coder0.properties = byteArray0;
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(byteArrayInputStream0, coder0, byteArray0);
      Coders.LZMADecoder coders_LZMADecoder0 = new Coders.LZMADecoder();
      try { 
        coders_LZMADecoder0.decode(inputStream0, coder0, coder0.properties);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // javax.crypto.IllegalBlockSizeException: Input length not multiple of 16 bytes
         //
         verifyException("javax.crypto.CipherInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(byteArrayInputStream0, coder0, byteArray0);
      ObjectInputStream objectInputStream0 = null;
      try {
        objectInputStream0 = new ObjectInputStream(inputStream0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$AES256SHA256Decoder$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[20];
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      Coder coder0 = new Coder();
      try { 
        coders_BZIP2Decoder0.decode((InputStream) null, coder0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No InputStream
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      SevenZMethod sevenZMethod0 = SevenZMethod.BZIP2;
      // Undeclared exception!
      try { 
        Coders.addEncoder((OutputStream) null, sevenZMethod0, byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      SevenZMethod sevenZMethod0 = SevenZMethod.DEFLATE;
      // Undeclared exception!
      try { 
        Coders.addEncoder((OutputStream) null, sevenZMethod0, byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.zip.DeflaterOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      Coder coder0 = new Coder();
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0, 279, (-168));
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

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      SevenZMethod sevenZMethod0 = SevenZMethod.COPY;
      OutputStream outputStream0 = Coders.addEncoder((OutputStream) null, sevenZMethod0, byteArray0);
      assertNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      coder0.decompressionMethodId = byteArray0;
      InputStream inputStream0 = Coders.addDecoder(byteArrayInputStream0, coder0, byteArray0);
      assertSame(inputStream0, byteArrayInputStream0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SevenZMethod sevenZMethod0 = SevenZMethod.BZIP2;
      Coders.CopyDecoder coders_CopyDecoder0 = new Coders.CopyDecoder();
      Coders.CoderId coders_CoderId0 = new Coders.CoderId(sevenZMethod0, coders_CopyDecoder0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
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
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      try { 
        Coders.addDecoder(byteArrayInputStream0, coder0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unsupported compression method null
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      Coder coder0 = new Coder();
      InputStream inputStream0 = coders_DeflateDecoder0.decode(byteArrayInputStream0, coder0, byteArray0);
      ObjectInputStream objectInputStream0 = null;
      try {
        objectInputStream0 = new ObjectInputStream(inputStream0);
        fail("Expecting exception: ZipException");
      
      } catch(Throwable e) {
         //
         // invalid stored block lengths
         //
         verifyException("java.util.zip.InflaterInputStream", e);
      }
  }
}