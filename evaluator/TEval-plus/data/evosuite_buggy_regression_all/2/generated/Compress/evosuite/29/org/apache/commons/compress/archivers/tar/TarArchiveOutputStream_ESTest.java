/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:48:31 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PipedOutputStream;
import java.io.UnsupportedEncodingException;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      tarArchiveOutputStream0.getCount();
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("1i|q@u3@", true);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "1i|q@u3@", map0);
      assertEquals(1536, tarArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("*mZWv[}*E");
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockFileOutputStream0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(bufferedOutputStream0, 1000);
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("last modification time");
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockPrintStream0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(bufferedOutputStream0, (-1732), "Uy");
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(1024, tarArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      File file0 = MockFile.createTempFile("BLKDEV", "BLKDEV");
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, "BLKDEV");
      tarArchiveOutputStream0.setAddPaxHeadersForNonAsciiNames(false);
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(13);
      assertEquals(512, tarArchiveOutputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      File file0 = MockFile.createTempFile(")CVumocCJJuXL", "", (File) mockFile0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      BufferedOutputStream bufferedOutputStream0 = new BufferedOutputStream(mockPrintStream0, 66);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(bufferedOutputStream0, 17, 17);
      tarArchiveOutputStream0.flush();
      assertEquals(17, tarArchiveOutputStream0.getRecordSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      File file0 = MockFile.createTempFile("BLKDEV", "BLKDEV");
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, "BLKDEV");
      tarArchiveOutputStream0.close();
      try { 
        tarArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archive has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("1i|q@u3@", true);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("*mZWv[}*E");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("9`4=o_Z");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("qzc `8Md30}1&");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      tarArchiveOutputStream0.finish();
      try { 
        tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "9`4=o_Z", hashMap0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      MockFile mockFile0 = new MockFile("Q\"~gkhHL\"knZ|u]M>Q+");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("1i|q@u3@");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      tarArchiveOutputStream0.setBigNumberMode(1);
      tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "Q\"~gkhHL\"knZ|u]M>Q+", hashMap0);
      assertEquals(512, tarArchiveOutputStream0.getCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("1i|q@u3@");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      tarArchiveOutputStream0.close();
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("1i|q@u3@");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // No current entry to close
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.write((byte[]) null, (-856), (-856));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No current tar entry
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("1i|q@u3@", true);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, true);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      try { 
        tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "", map0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '637' bytes exceeds size in header of '0' bytes for entry './PaxHeaders.X/'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("f@A4Mhst-5hc_0ip");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("", "");
      try { 
        tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "", hashMap0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("9`4=o_Z");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("sL5/0LjYSw-YVibg");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "sL5/0LjYSw-YVibg", hashMap0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, (-2180), 33188);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.writePaxHeaders((TarArchiveEntry) null, "ustar\u0000", hashMap0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      File file0 = MockFile.createTempFile("BLKDEV", "BLKDEV");
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, "BLKDEV");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", true);
      Map<String, String> map0 = ZoneId.SHORT_IDS;
      try { 
        tarArchiveOutputStream0.writePaxHeaders(tarArchiveEntry0, "&Yk+.l0BEx3gBVZjc", map0);
        fail("Expecting exception: UnsupportedEncodingException");
      
      } catch(UnsupportedEncodingException e) {
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      File file0 = MockFile.createTempFile("BLKDEV", "BLKDEV");
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, "BLKDEV");
      tarArchiveOutputStream0.createArchiveEntry(file0, "BLKDEV");
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      File file0 = MockFile.createTempFile("BLKDEV", "BLKDEV");
      MockPrintStream mockPrintStream0 = new MockPrintStream(file0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, "BLKDEV");
      tarArchiveOutputStream0.close();
      try { 
        tarArchiveOutputStream0.createArchiveEntry(file0, "BLKDEV");
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream has already been finished
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.arj.ArjArchiveEntry");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 3, 2);
      tarArchiveOutputStream0.finish();
  }
}