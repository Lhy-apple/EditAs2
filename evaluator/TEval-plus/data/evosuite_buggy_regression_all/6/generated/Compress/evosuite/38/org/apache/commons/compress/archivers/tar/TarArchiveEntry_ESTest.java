/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:14:07 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.File;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.zip.ZipEncoding;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveEntry_ESTest extends TarArchiveEntry_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" exceeds maximum signed long", true);
      String string0 = tarArchiveEntry0.getUserName();
      assertEquals(" exceeds maximum signed long", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = null;
      try {
        tarArchiveEntry0 = new TarArchiveEntry((byte[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      File file0 = MockFile.createTempFile("a{3V1:oy[", (String) null);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0);
      long long0 = tarArchiveEntry0.getSize();
      assertEquals(0L, long0);
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals("tmp/a{3V1:oy[0.tmp", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("]= ldoGBfuAN-");
      int int0 = tarArchiveEntry0.getMode();
      assertEquals(33188, int0);
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals("]= ldoGBfuAN-", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("C1*=']f8\u0006%");
      tarArchiveEntry0.setIds(16877, 16877);
      assertEquals(16877, tarArchiveEntry0.getUserId());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("z");
      boolean boolean0 = tarArchiveEntry0.isSparse();
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals("z", tarArchiveEntry0.getName());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(".MQcilvt/", (byte)98);
      boolean boolean0 = tarArchiveEntry0.isFile();
      assertFalse(boolean0);
      assertEquals(16877, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isGlobalPaxHeader());
      assertFalse(tarArchiveEntry0.isOldGNUSparse());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      tarArchiveEntry0.setLinkName(" byte field.");
      assertEquals(" byte field.", tarArchiveEntry0.getLinkName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("z");
      tarArchiveEntry0.setNames("z", "z");
      assertEquals("z", tarArchiveEntry0.getGroupName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("ustar\u0000", (byte)52, false);
      tarArchiveEntry0.setName("ustar\u0000");
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals("ustar\u0000", tarArchiveEntry0.getName());
      assertTrue(tarArchiveEntry0.isBlockDevice());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("");
      Date date0 = tarArchiveEntry0.getLastModifiedDate();
      assertEquals("", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals("Fri Feb 14 20:21:21 GMT 2014", date0.toString());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertFalse(tarArchiveEntry0.isLink());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockFile mockFile0 = new MockFile("US-ASCII", "US-ASCII");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      int int0 = tarArchiveEntry0.getDevMajor();
      assertEquals("data/lhy/TEval-plus/US-ASCII/US-ASCII", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("os.name");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.writeEntryHeader((byte[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      tarArchiveEntry0.isCheckSumOK();
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(" byte field.", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getGroupId());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      tarArchiveEntry0.hashCode();
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(" byte field.", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertFalse(tarArchiveEntry0.isDirectory());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("r", false);
      tarArchiveEntry0.isDescendent(tarArchiveEntry0);
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals("r", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isDirectory());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("_|SPAA`8Q1#'", (byte) (-39));
      int int0 = tarArchiveEntry0.getGroupId();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals("_|SPAA`8Q1#'", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isPaxHeader());
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(tarArchiveEntry0.isGNULongLinkEntry());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0, int0);
      assertFalse(tarArchiveEntry0.isCharacterDevice());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("");
      long long0 = tarArchiveEntry0.getLongGroupId();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0L, long0);
      assertEquals("", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("GW^S#h", (byte) (-86));
      tarArchiveEntry0.isExtended();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals("GW^S#h", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isBlockDevice());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(tarArchiveEntry0.isPaxHeader());
      assertFalse(tarArchiveEntry0.isGNULongLinkEntry());
      assertFalse(tarArchiveEntry0.isCharacterDevice());
      assertEquals(0, tarArchiveEntry0.getGroupId());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("(P:wWP=RJ'");
      long long0 = tarArchiveEntry0.getRealSize();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0L, long0);
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals("(P:wWP=RJ'", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isDirectory());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("netware");
      String string0 = tarArchiveEntry0.getLinkName();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals("netware", tarArchiveEntry0.getName());
      assertEquals("", string0);
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("windows");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.setModTime((Date) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      TarArchiveEntry tarArchiveEntry0 = null;
      try {
        tarArchiveEntry0 = new TarArchiveEntry(byteArray0, (ZipEncoding) null);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 99
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("");
      boolean boolean0 = tarArchiveEntry0.equals((Object) tarArchiveEntry0);
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals("", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("X*q)l");
      int int0 = tarArchiveEntry0.getDevMinor();
      assertEquals("X*q)l", tarArchiveEntry0.getName());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("X*q)l");
      int int0 = tarArchiveEntry0.getUserId();
      assertEquals(0, int0);
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals("X*q)l", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("m/L`Vi$VU:{", (byte)90);
      String string0 = tarArchiveEntry0.getGroupName();
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isCharacterDevice());
      assertFalse(tarArchiveEntry0.isSymbolicLink());
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
      assertEquals("m/L`Vi$VU:{", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isGlobalPaxHeader());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals("", string0);
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("os.name");
      assertEquals(33188, tarArchiveEntry0.getMode());
      
      tarArchiveEntry0.setMode(16877);
      assertEquals(0, tarArchiveEntry0.getDevMinor());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("Size is out of range: ");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.fillGNUSparse1xData((Map<String, String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      File file0 = MockFile.createTempFile("GW^S#h ", "GW^S#h ");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(file0, "GW^S#h ");
      long long0 = tarArchiveEntry0.getLongUserId();
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals("GW^S#h ", tarArchiveEntry0.getName());
      assertEquals(0L, long0);
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("J7zpP5LA0}~JO\"{G 9/");
      tarArchiveEntry0.setModTime((long) 1000);
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(tarArchiveEntry0.isFile());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(16877, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("X*q)l");
      tarArchiveEntry0.getFile();
      assertEquals("X*q)l", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0, tarArchiveEntry0.getUserId());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("L)cRR^~)", (byte)76);
      tarArchiveEntry0.isFile();
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals("L)cRR^~)", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertTrue(tarArchiveEntry0.isGNULongNameEntry());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(33188, tarArchiveEntry0.getMode());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, ":X+B[/HrXe3Ya/");
      assertEquals(":X+B[/HrXe3Ya/", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(16877, tarArchiveEntry0.getMode());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveEntry0.getDirectoryEntries();
      assertEquals("/", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      tarArchiveEntry0.equals((Object) null);
      assertEquals(" byte field.", tarArchiveEntry0.getName());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertFalse(tarArchiveEntry0.isLink());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      boolean boolean0 = tarArchiveEntry0.equals((Object) " byte field.");
      assertFalse(boolean0);
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(" byte field.", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("\"", (byte)87);
      tarArchiveEntry0.setSize((byte)87);
      assertEquals(87L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("[z,lwi&Yl~3b eqLO");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.setSize((byte) (-46));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Size is out of range: -46
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("os.name");
      tarArchiveEntry0.setDevMajor(1000);
      assertEquals(1000, tarArchiveEntry0.getDevMajor());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("org.apache.commons.compress.archivers.tar.TarUtils");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.setDevMajor((-2004318069));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Major device number is out of range: -2004318069
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("[z,lwi&Yl~3b eqLO");
      tarArchiveEntry0.setDevMinor(31);
      assertEquals(31, tarArchiveEntry0.getDevMinor());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("e3UzV@)A=]");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.setDevMinor((-6254807));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Minor device number is out of range: -6254807
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveEntry", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("u^}y.bN,fS(V///", (byte) (-41), false);
      boolean boolean0 = tarArchiveEntry0.isGNULongLinkEntry();
      assertFalse(tarArchiveEntry0.isCharacterDevice());
      assertFalse(tarArchiveEntry0.isPaxHeader());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(boolean0);
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
      assertFalse(tarArchiveEntry0.isFile());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(16877, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isSymbolicLink());
      assertFalse(tarArchiveEntry0.isFIFO());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("m/L`Vi$VU:{", (byte)75);
      boolean boolean0 = tarArchiveEntry0.isGNULongLinkEntry();
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertTrue(boolean0);
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals("m/L`Vi$VU:{", tarArchiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      boolean boolean0 = tarArchiveEntry0.isGNULongNameEntry();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(" byte field.", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertFalse(boolean0);
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0, tarArchiveEntry0.getGroupId());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("m/L`Vi$VU:{", (byte)76);
      boolean boolean0 = tarArchiveEntry0.isGNULongNameEntry();
      assertEquals("m/L`Vi$VU:{", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertTrue(boolean0);
      assertEquals(0L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("rc7R2PVg\"", (byte)120);
      boolean boolean0 = tarArchiveEntry0.isPaxHeader();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertTrue(boolean0);
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals("rc7R2PVg\"", tarArchiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("\"", (byte)87);
      boolean boolean0 = tarArchiveEntry0.isPaxHeader();
      assertFalse(tarArchiveEntry0.isCharacterDevice());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertFalse(boolean0);
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals("\"", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isGNULongLinkEntry());
      assertEquals(0L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("\"", (byte)88);
      boolean boolean0 = tarArchiveEntry0.isPaxHeader();
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertTrue(boolean0);
      assertEquals("\"", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getSize());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("os.name");
      boolean boolean0 = tarArchiveEntry0.isGlobalPaxHeader();
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertFalse(boolean0);
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals("os.name", tarArchiveEntry0.getName());
      assertEquals(33188, tarArchiveEntry0.getMode());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)103, false);
      boolean boolean0 = tarArchiveEntry0.isGlobalPaxHeader();
      assertTrue(boolean0);
      assertEquals("", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals(33188, tarArchiveEntry0.getMode());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("u^}y.bN,fS(V///", (byte) (-41), false);
      boolean boolean0 = tarArchiveEntry0.isDirectory();
      assertTrue(boolean0);
      assertFalse(tarArchiveEntry0.isPaxHeader());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(16877, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      MockFile mockFile0 = new MockFile("^");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveEntry0.isDirectory();
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals("^", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0, tarArchiveEntry0.getGroupId());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("u^}y.bN,fS(V///");
      boolean boolean0 = tarArchiveEntry0.isDirectory();
      assertTrue(boolean0);
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertFalse(tarArchiveEntry0.isFile());
      assertEquals(16877, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0, tarArchiveEntry0.getUserId());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(" byte field.");
      boolean boolean0 = tarArchiveEntry0.isDirectory();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertFalse(boolean0);
      assertEquals(" byte field.", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getUserId());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      MockFile mockFile0 = new MockFile("k");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveEntry0.isFile();
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals("k", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0, tarArchiveEntry0.getUserId());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("Pl(=!{xT.]3}ap#E", (byte)0);
      tarArchiveEntry0.isFile();
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals("Pl(=!{xT.]3}ap#E", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isLink());
      assertFalse(tarArchiveEntry0.isCharacterDevice());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertFalse(tarArchiveEntry0.isGNULongLinkEntry());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(tarArchiveEntry0.isGNUSparse());
      assertFalse(tarArchiveEntry0.isFIFO());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertFalse(tarArchiveEntry0.isGlobalPaxHeader());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("^");
      tarArchiveEntry0.isFile();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals("^", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertFalse(tarArchiveEntry0.isLink());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertFalse(tarArchiveEntry0.isDirectory());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)54);
      boolean boolean0 = tarArchiveEntry0.isSymbolicLink();
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals("", tarArchiveEntry0.getName());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertTrue(tarArchiveEntry0.isFIFO());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("5+y_$Z'ID8M$", (byte)50, true);
      boolean boolean0 = tarArchiveEntry0.isSymbolicLink();
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals("5+y_$Z'ID8M$", tarArchiveEntry0.getName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("HxL<45 F`a`j^", (byte)49, false);
      boolean boolean0 = tarArchiveEntry0.isLink();
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertTrue(boolean0);
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals("HxL<45 F`a`j^", tarArchiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("t");
      boolean boolean0 = tarArchiveEntry0.isCharacterDevice();
      assertEquals("t", tarArchiveEntry0.getName());
      assertFalse(tarArchiveEntry0.isDirectory());
      assertFalse(boolean0);
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("IBM850", (byte)51, true);
      boolean boolean0 = tarArchiveEntry0.isCharacterDevice();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertTrue(boolean0);
      assertEquals("IBM850", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("");
      boolean boolean0 = tarArchiveEntry0.isBlockDevice();
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertFalse(boolean0);
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals("", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getUserId());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("ustar\u0000", (byte)52, false);
      boolean boolean0 = tarArchiveEntry0.isBlockDevice();
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals("ustar\u0000", tarArchiveEntry0.getName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(";s.name");
      boolean boolean0 = tarArchiveEntry0.isFIFO();
      assertFalse(tarArchiveEntry0.isDirectory());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(";s.name", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)54);
      boolean boolean0 = tarArchiveEntry0.isFIFO();
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals("", tarArchiveEntry0.getName());
      assertTrue(boolean0);
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals(0, tarArchiveEntry0.getUserId());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("SCHILY.realsize", (byte)83, false);
      boolean boolean0 = tarArchiveEntry0.isSparse();
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertTrue(boolean0);
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals("SCHILY.realsize", tarArchiveEntry0.getName());
      assertEquals(0L, tarArchiveEntry0.getLongGroupId());
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("(^|}xE^KK9WA\"");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      tarArchiveEntry0.fillStarSparseData(hashMap0);
      boolean boolean0 = tarArchiveEntry0.isSparse();
      assertTrue(tarArchiveEntry0.isStarSparse());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("", (byte)54);
      tarArchiveEntry0.getDirectoryEntries();
      assertEquals(0L, tarArchiveEntry0.getSize());
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertTrue(tarArchiveEntry0.isFIFO());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(33188, tarArchiveEntry0.getMode());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
      assertEquals("", tarArchiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      MockFile mockFile0 = new MockFile("Length ");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveEntry0.getDirectoryEntries();
      assertEquals(0, tarArchiveEntry0.getGroupId());
      assertEquals("Length ", tarArchiveEntry0.getName());
      assertEquals(0, tarArchiveEntry0.getDevMajor());
      assertEquals(0, tarArchiveEntry0.getDevMinor());
      assertEquals(0L, tarArchiveEntry0.getLongUserId());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("J7zpP5LA0}~JO\"{G 9/");
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      hashMap0.put("SCHILY.realsize", "+Lsd8bXh{W/");
      // Undeclared exception!
      try { 
        tarArchiveEntry0.fillStarSparseData(hashMap0);
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"+Lsd8bXh{W/\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }
}