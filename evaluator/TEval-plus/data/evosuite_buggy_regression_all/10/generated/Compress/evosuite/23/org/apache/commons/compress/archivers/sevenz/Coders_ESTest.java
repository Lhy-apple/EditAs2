/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:09:39 GMT 2023
 */

package org.apache.commons.compress.archivers.sevenz;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.util.Enumeration;
import java.util.zip.ZipException;
import org.apache.commons.compress.archivers.sevenz.Coder;
import org.apache.commons.compress.archivers.sevenz.Coders;
import org.apache.commons.compress.archivers.sevenz.SevenZMethod;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
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
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false, false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      byte[] byteArray0 = new byte[5];
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      coder0.properties = byteArray0;
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(sequenceInputStream0, coder0, byteArray0);
      Coders.LZMADecoder coders_LZMADecoder0 = new Coders.LZMADecoder();
      SequenceInputStream sequenceInputStream1 = new SequenceInputStream(sequenceInputStream0, inputStream0);
      try { 
        coders_LZMADecoder0.decode(sequenceInputStream1, coder0, byteArray0);
        fail("Expecting exception: EOFException");
      
      } catch(EOFException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.io.DataInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      Coder coder0 = new Coder();
      try { 
        coders_BZIP2Decoder0.decode(byteArrayInputStream0, coder0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream is not in the BZip2 format
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Coders.BZIP2Decoder coders_BZIP2Decoder0 = new Coders.BZIP2Decoder();
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        coders_BZIP2Decoder0.encode((OutputStream) null, byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      byte[] byteArray0 = new byte[2];
      // Undeclared exception!
      try { 
        coders_DeflateDecoder0.encode((OutputStream) null, byteArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.zip.DeflaterOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Coder coder0 = new Coder();
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(15);
      SevenZMethod sevenZMethod0 = SevenZMethod.COPY;
      OutputStream outputStream0 = Coders.addEncoder(byteArrayOutputStream0, sevenZMethod0, coder0.properties);
      assertSame(outputStream0, byteArrayOutputStream0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Coders.CopyDecoder coders_CopyDecoder0 = new Coders.CopyDecoder();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      coder0.decompressionMethodId = byteArray0;
      InputStream inputStream0 = Coders.addDecoder(byteArrayInputStream0, coder0, byteArray0);
      assertSame(byteArrayInputStream0, inputStream0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SevenZMethod sevenZMethod0 = SevenZMethod.LZMA;
      Coders.CoderId coders_CoderId0 = new Coders.CoderId(sevenZMethod0, (Coders.CoderBase) null);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Coder coder0 = new Coder();
      try { 
        Coders.addDecoder((InputStream) null, coder0, coder0.properties);
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
      byte[] byteArray0 = new byte[8];
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream((byte)0);
      Coders.CoderId[] coders_CoderIdArray0 = new Coders.CoderId[0];
      Coders.coderTable = coders_CoderIdArray0;
      SevenZMethod sevenZMethod0 = SevenZMethod.LZMA2;
      try { 
        Coders.addEncoder(byteArrayOutputStream0, sevenZMethod0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Unsupported compression method LZMA2
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("LZMA2");
      SevenZMethod sevenZMethod0 = SevenZMethod.AES256SHA256;
      byte[] byteArray0 = new byte[2];
      // Undeclared exception!
      try { 
        Coders.addEncoder(mockPrintStream0, sevenZMethod0, byteArray0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // method doesn't support writing
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$CoderBase", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      Coder coder0 = new Coder();
      byte[] byteArray0 = new byte[5];
      Coder coder1 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(sequenceInputStream0, coder1, byteArray0);
      InputStream inputStream1 = coders_DeflateDecoder0.decode(inputStream0, coder0, byteArray0);
      coder1.properties = byteArray0;
      ObjectInputStream objectInputStream0 = null;
      try {
        objectInputStream0 = new ObjectInputStream(inputStream1);
        fail("Expecting exception: EOFException");
      
      } catch(Throwable e) {
         //
         // Unexpected end of ZLIB input stream
         //
         verifyException("java.util.zip.InflaterInputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      byte[] byteArray0 = new byte[9];
      byteArray0[1] = (byte) (-66);
      coder0.properties = byteArray0;
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(sequenceInputStream0, coder0, coder0.properties);
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
  public void test14()  throws Throwable  {
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      byte[] byteArray0 = new byte[5];
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      coder0.properties = byteArray0;
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(sequenceInputStream0, coder0, (byte[]) null);
      Coders.LZMADecoder coders_LZMADecoder0 = new Coders.LZMADecoder();
      try { 
        coders_LZMADecoder0.decode(inputStream0, coder0, byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Cannot read encrypted files without a password
         //
         verifyException("org.apache.commons.compress.archivers.sevenz.Coders$AES256SHA256Decoder$1", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      Coder coder0 = new Coder();
      Coders.AES256SHA256Decoder coders_AES256SHA256Decoder0 = new Coders.AES256SHA256Decoder();
      byte[] byteArray0 = new byte[9];
      byteArray0[0] = (byte) (-66);
      coder0.properties = byteArray0;
      InputStream inputStream0 = coders_AES256SHA256Decoder0.decode(sequenceInputStream0, coder0, coder0.properties);
      ObjectInputStream objectInputStream0 = new ObjectInputStream(inputStream0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ByteArrayInputStream byteArrayInputStream0 = new ByteArrayInputStream(byteArray0);
      Coder coder0 = new Coder();
      Coders.DeflateDecoder coders_DeflateDecoder0 = new Coders.DeflateDecoder();
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(byteArrayInputStream0);
      InputStream inputStream0 = coders_DeflateDecoder0.decode(pushbackInputStream0, coder0, byteArray0);
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