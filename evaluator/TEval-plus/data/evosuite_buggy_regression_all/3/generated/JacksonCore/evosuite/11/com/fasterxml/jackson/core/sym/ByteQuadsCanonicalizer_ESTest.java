/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:34:52 GMT 2023
 */

package com.fasterxml.jackson.core.sym;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ByteQuadsCanonicalizer_ESTest extends ByteQuadsCanonicalizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1674));
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 58, 58);
      int[] intArray0 = new int[10];
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(110, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int int0 = byteQuadsCanonicalizer0.hashSeed();
      assertEquals(839877741, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(0);
      int int0 = byteQuadsCanonicalizer0.bucketCount();
      assertEquals(0, byteQuadsCanonicalizer0.hashSeed());
      assertEquals(0, int0);
      assertEquals(0, byteQuadsCanonicalizer0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1.addName((String) null, 440, 16, 244);
      byteQuadsCanonicalizer1.release();
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0.release();
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1.release();
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertFalse(byteQuadsCanonicalizer1.maybeDirty());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertTrue(byteQuadsCanonicalizer0.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(351);
      byteQuadsCanonicalizer1._count = (-1);
      byteQuadsCanonicalizer1.addName("*+Gr=*", 351);
      byteQuadsCanonicalizer1.release();
      assertEquals(0, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(64);
      int int0 = byteQuadsCanonicalizer1.size();
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertFalse(byteQuadsCanonicalizer1.maybeDirty());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(0, int0);
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int int0 = byteQuadsCanonicalizer0.size();
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(444);
      byteQuadsCanonicalizer1.addName("i-=8T'cG", 444, 444);
      byteQuadsCanonicalizer1.addName("i-=8T'cG", 444, 444);
      byteQuadsCanonicalizer1.toString();
      assertEquals(2, byteQuadsCanonicalizer1.totalCount());
      assertEquals(1, byteQuadsCanonicalizer1.secondaryCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 444;
      int[] intArray0 = new int[5];
      intArray0[3] = 444;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.toString();
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(243);
      int[] intArray0 = new int[9];
      byteQuadsCanonicalizer1.findName(intArray0, (-1));
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(280);
      byteQuadsCanonicalizer1.addName("!le|", 280);
      byteQuadsCanonicalizer1.findName(24);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(391);
      byteQuadsCanonicalizer1.addName("", 391);
      byteQuadsCanonicalizer1.findName(391);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(243);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=1, 0/0/0/110 pri/sec/ter/spill (=0), total:110]", 243, 243, 1);
      int[] intArray0 = new int[9];
      byteQuadsCanonicalizer1.findName(intArray0, (-1));
      assertEquals(1, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(243);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=1, 0/0/0/110 pri/sec/ter/spill (=0), total:110]", 243, 243, 1);
      int[] intArray0 = new int[9];
      intArray0[0] = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=1, 0/0/0/110 pri/sec/ter/spill (=0), total:110]", intArray0, 1);
      byteQuadsCanonicalizer1.findName(intArray0, (-1));
      assertEquals(2, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 473, 39, 65599);
      byteQuadsCanonicalizer1.addName("", 473);
      byteQuadsCanonicalizer1.findName(37);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      int[] intArray0 = new int[37];
      intArray0[3] = 3936;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(intArray0, 0);
      assertEquals((-7), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1.findName(244, 244);
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.findName(244, 1);
      assertEquals(1, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(2);
      byteQuadsCanonicalizer1.addName("[%s: size=%d, hashSize=%d, %d/%d/%d/%d pri/sec/ter/spill (=%s), total:%d]", 2, 2);
      byteQuadsCanonicalizer1.findName(2, 2);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      int[] intArray0 = new int[8];
      intArray0[3] = 2840;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(4, 2840);
      assertEquals((-7), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 440, 16, 244);
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.findName(244, 440);
      assertEquals(2, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 21, 16, 65599);
      byteQuadsCanonicalizer1.addName((String) null, 21, 21);
      byteQuadsCanonicalizer1.findName(244, 244);
      assertEquals(2, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 244, 16, 244);
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.findName(244, 244);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1.findName(244, 244, 1);
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(243);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 16, 16, 1);
      byteQuadsCanonicalizer1.findName(16, 1, 1);
      assertEquals(1, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(243);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 16, 16, 1);
      byteQuadsCanonicalizer1.findName(16, 16, 16);
      assertEquals(1, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-8));
      byteQuadsCanonicalizer1.addName("d6ts^d2-;*jTFJ;3", (-8), (-8), (-8));
      byteQuadsCanonicalizer1.findName((-8), (-8), (-8));
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.findName(244, 244, 1);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.addName((String) null, 244, 244, 1);
      byteQuadsCanonicalizer1.findName(244, 244, 1);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 58, 58);
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 58, 58, 1);
      byteQuadsCanonicalizer1.findName(58, 2952, 2952);
      assertEquals(2, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(273);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 273, 273);
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 16, 16, 1);
      byteQuadsCanonicalizer1.findName(16, 16, 16);
      assertEquals(2, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 440, 16, 244);
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.findName(16, 16, 1);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1674));
      int[] intArray0 = new int[10];
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(839877741, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1658));
      int[] intArray0 = new int[10];
      byteQuadsCanonicalizer1.addName((String) null, intArray0, 7);
      byteQuadsCanonicalizer1.findName(intArray0, 7);
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1674));
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 58, 58);
      int[] intArray0 = new int[7];
      intArray0[1] = (-1674);
      byteQuadsCanonicalizer1.addName((String) null, intArray0, 4);
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      byteQuadsCanonicalizer0._spilloverEnd = 3936;
      int[] intArray0 = new int[37];
      intArray0[3] = 3936;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 39
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 244, 244);
      byteQuadsCanonicalizer1.addName((String) null, 244, 244, 1);
      int[] intArray0 = new int[6];
      byteQuadsCanonicalizer1.findName(intArray0, 2);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 61, 61);
      byteQuadsCanonicalizer1.addName((String) null, 61, 61, 1);
      int[] intArray0 = new int[7];
      intArray0[0] = 244;
      intArray0[1] = 244;
      intArray0[2] = 1;
      intArray0[3] = 2;
      byteQuadsCanonicalizer1.addName((String) null, intArray0, 4);
      byteQuadsCanonicalizer1.addName((String) null, 1, 190);
      byteQuadsCanonicalizer1.findName(1, 1767);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      byteQuadsCanonicalizer0._spilloverEnd = 2840;
      int[] intArray0 = new int[8];
      intArray0[3] = 2840;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(4, 2840);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 28
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(244);
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 58, 58);
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 58, 58, 1);
      int[] intArray0 = new int[6];
      byteQuadsCanonicalizer1.findName(intArray0, 3);
      assertEquals(2, byteQuadsCanonicalizer1.totalCount());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      int[] intArray0 = new int[4];
      intArray0[0] = 4;
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      assertEquals((-7), byteQuadsCanonicalizer0.spilloverCount());
      
      byteQuadsCanonicalizer0.findName(4, 359, 1690);
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      byteQuadsCanonicalizer0._spilloverEnd = 3875;
      int[] intArray0 = new int[11];
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(3875, 4, 4);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 28
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      int[] intArray0 = new int[37];
      intArray0[0] = 4;
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(intArray0, 4);
      assertEquals((-7), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1674));
      byteQuadsCanonicalizer1._hashSize = 1;
      byteQuadsCanonicalizer1.addName((String) null, 58, 58);
      byteQuadsCanonicalizer1.addName((String) null, 58, 58, 1);
      int[] intArray0 = new int[10];
      intArray0[0] = 2;
      intArray0[1] = (-1674);
      byteQuadsCanonicalizer1.addName((String) null, intArray0, 4);
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(4, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      byteQuadsCanonicalizer0._spilloverEnd = 3936;
      int[] intArray0 = new int[37];
      intArray0[0] = 4;
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 4);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 40
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("i-=8T'cG", 473, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      int[] intArray0 = new int[11];
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 14, 14);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4;
      byteQuadsCanonicalizer0._spilloverEnd = 3905;
      int[] intArray0 = new int[11];
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", 14, 14);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 3905
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("dm^SD5 &QmV2.C2O:", (int[]) null, (-8));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(391);
      byteQuadsCanonicalizer1._count = 391;
      byteQuadsCanonicalizer1.addName("", 391);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("", 391, 391);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=392, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1674));
      byteQuadsCanonicalizer1._hashSize = 1;
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer1.addName((String) null, intArray0, 4);
      byteQuadsCanonicalizer1.addName("(47vC15 \"*3f{ME", 4, 4, (-1674));
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(415);
      byteQuadsCanonicalizer1._count = 415;
      byteQuadsCanonicalizer1.addName("f", 415);
      int[] intArray0 = new int[6];
      intArray0[3] = 1560;
      byteQuadsCanonicalizer1._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("f", 415, 415);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(346);
      byteQuadsCanonicalizer0._hashSize = 2713;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0._reportTooManyCollisions();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Spill-over slots in symbol table with 0 entries, hash area of 2713 slots is now full (all 339 slots -- suspect a DoS attack based on hash collisions. You can disable the check via `JsonFactory.Feature.FAIL_ON_SYMBOL_HASH_OVERFLOW`
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(415);
      byteQuadsCanonicalizer1._count = 415;
      byteQuadsCanonicalizer1.addName("f", 415);
      byteQuadsCanonicalizer1._hashSize = 1560;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName("f", 415, 415);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 3104
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(503);
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(6000);
      assertEquals(7, int0);
  }
}