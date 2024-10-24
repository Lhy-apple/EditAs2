/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:21:03 GMT 2023
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
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(0);
      byteQuadsCanonicalizer1.addName("6q%%", (-727), (-727), (-727));
      byteQuadsCanonicalizer1.findName(1426, 0, 1426);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int int0 = byteQuadsCanonicalizer0.hashSeed();
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(839877741, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0.bucketCount();
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-3213));
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", (-3213));
      byteQuadsCanonicalizer1.release();
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0.release();
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-4274));
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-4274));
      byteQuadsCanonicalizer1.release();
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals((-4274), byteQuadsCanonicalizer1.hashSeed());
      assertFalse(byteQuadsCanonicalizer1.maybeDirty());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(65599);
      byteQuadsCanonicalizer1._count = 65599;
      byteQuadsCanonicalizer1.addName(" 68", 65599, 65599, 65600);
      byteQuadsCanonicalizer1.release();
      assertEquals(65600, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(3337);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(3337);
      int int0 = byteQuadsCanonicalizer1.size();
      assertEquals(3337, byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, int0);
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int int0 = byteQuadsCanonicalizer0.size();
      assertEquals(0, int0);
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1008));
      byteQuadsCanonicalizer1.addName(",;-M7(@g0bHiJeY<", (-1008));
      byteQuadsCanonicalizer1.addName(",;-M7(@g0bHiJeY<", (-1008));
      byteQuadsCanonicalizer1.addName(",;-M7(@g0bHiJeY<", (-1008), (-1008), (-1008));
      byteQuadsCanonicalizer1.toString();
      assertTrue(byteQuadsCanonicalizer1.maybeDirty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[5];
      byteQuadsCanonicalizer0._hashSize = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(intArray0, 1);
      assertEquals(0, byteQuadsCanonicalizer0.size());
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-3213));
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", (-3213));
      byteQuadsCanonicalizer1.findName(447);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-999));
      byteQuadsCanonicalizer1.addName(" entriesV hash area of ", (-999));
      byteQuadsCanonicalizer1.findName((-999));
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[5];
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0._hashSize = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(intArray0, 1);
      assertEquals((-2), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-3900));
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", (-1008), 16);
      byteQuadsCanonicalizer1.findName((-1008), (-1008));
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-999));
      byteQuadsCanonicalizer1.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=64, 0/0/0/0 pri/sec/ter/spill (=0), total:0]", (-999), (-999));
      byteQuadsCanonicalizer1.findName((-999), (-999));
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-4274));
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-4274));
      byteQuadsCanonicalizer1.findName((-4274), (-4274));
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals((-4274), byteQuadsCanonicalizer1.hashSeed());
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-999));
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", (-999), (-999));
      byteQuadsCanonicalizer1.addName("Failed rehash(): old count=", (-999), (-999));
      byteQuadsCanonicalizer1.findName((-2503), (-2503));
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[5];
      byteQuadsCanonicalizer0._hashSize = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName((-1), 0, (-1041));
      assertEquals(839877741, byteQuadsCanonicalizer0.hashSeed());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1008));
      byteQuadsCanonicalizer1.addName("N", (-1008), (-2290), 60);
      byteQuadsCanonicalizer1.findName((-1008), 52, (-1008));
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(2147);
      byteQuadsCanonicalizer1.addName("lp3]TALop", 2147, 2147, 2147);
      byteQuadsCanonicalizer1.findName(2147, 2147, 2147);
      assertEquals(1, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1008));
      byteQuadsCanonicalizer1.addName("$,*>ZA2'K=YR9[66T", (-1008));
      byteQuadsCanonicalizer1.addName("$,*>ZA2'K=YR9[66T", (-1008), (-1008), (-1008));
      byteQuadsCanonicalizer1.findName(37, (-1008), 443);
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1008));
      byteQuadsCanonicalizer1.addName("UN-K6r?(E}NBoi\u0007", (-1008));
      byteQuadsCanonicalizer1.addName("UN-K6r?(E}NBoi\u0007", (-1008), (-1008), (-1008));
      byteQuadsCanonicalizer1.findName((-1008), (-1008), (-1008));
      assertEquals(2, byteQuadsCanonicalizer1.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(0);
      int[] intArray0 = new int[4];
      byteQuadsCanonicalizer1.addName("com.faste'xml.jacksn.core.s{m.ByteQuadsCanonicalizer$Tablenfo", intArray0, 4);
      byteQuadsCanonicalizer1.findName(intArray0, 4);
      assertEquals(1, byteQuadsCanonicalizer1.primaryCount());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1));
      int[] intArray0 = new int[8];
      intArray0[0] = (-1);
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1));
      byteQuadsCanonicalizer1.findName(intArray0, 6);
      assertEquals(0, byteQuadsCanonicalizer1.size());
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
      assertEquals(64, byteQuadsCanonicalizer1.bucketCount());
      assertEquals((-1), byteQuadsCanonicalizer1.hashSeed());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1));
      int[] intArray0 = new int[8];
      intArray0[0] = (-1);
      byteQuadsCanonicalizer0._secondaryStart = 2;
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      assertEquals((-1), byteQuadsCanonicalizer0.hashSeed());
      
      byteQuadsCanonicalizer0.findName(intArray0, 4);
      assertEquals(0, byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-8));
      int[] intArray0 = new int[7];
      intArray0[3] = (-8);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      assertEquals((-8), byteQuadsCanonicalizer0.hashSeed());
      
      byteQuadsCanonicalizer0.findName((-8));
      assertEquals(0, byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-3));
      byteQuadsCanonicalizer0._tertiaryShift = (-3);
      int[] intArray0 = new int[8];
      intArray0[3] = (-3);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      String string0 = byteQuadsCanonicalizer0.findName((-3));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(674);
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer0._spilloverEnd = 674;
      intArray0[3] = 674;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(674);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(674);
      int[] intArray0 = new int[5];
      intArray0[0] = 674;
      byteQuadsCanonicalizer0._spilloverEnd = 674;
      intArray0[3] = 674;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(674);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[8];
      intArray0[0] = 12;
      intArray0[3] = 12;
      byteQuadsCanonicalizer0._hashSize = 12;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName(12, 3064);
      assertEquals((-21), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(4);
      byteQuadsCanonicalizer0._spilloverEnd = 2137099297;
      int[] intArray0 = new int[22];
      intArray0[0] = 4;
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 2);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 24
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[5];
      intArray0[0] = (-1);
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0._hashSize = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName((-1), 0, (-1041));
      assertEquals((-2), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[5];
      intArray0[0] = (-1);
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0._hashSize = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0.findName((-1), (-504), 3247);
      assertEquals((-2), byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1));
      int[] intArray0 = new int[8];
      intArray0[0] = (-1);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      assertEquals((-1), byteQuadsCanonicalizer0.hashSeed());
      
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0.findName(intArray0, 3);
      assertEquals(0, byteQuadsCanonicalizer0.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1008));
      byteQuadsCanonicalizer1.addName("$,*>ZAr'K=Y{Rp9[66T", (-1008));
      byteQuadsCanonicalizer1.addName("$,*>ZAr'K=Y{Rp9[66T", (-1008));
      byteQuadsCanonicalizer1.addName("$,*>ZAr'K=Y{Rp9[66T", (-1008), (-1008), (-1008));
      byteQuadsCanonicalizer1.findName((-1008), (-1008), (-1008));
      assertEquals(1, byteQuadsCanonicalizer1.secondaryCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1));
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer0._spilloverEnd = 495;
      intArray0[0] = (-1);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      intArray0[3] = (-1);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 3);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1));
      int[] intArray0 = new int[8];
      byteQuadsCanonicalizer0._tertiaryShift = 4;
      intArray0[0] = (-1);
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      String string0 = byteQuadsCanonicalizer0.findName(intArray0, 8);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-1));
      int[] intArray0 = new int[8];
      intArray0[0] = (-1);
      intArray0[3] = (-1);
      byteQuadsCanonicalizer0._hashArea = intArray0;
      byteQuadsCanonicalizer0._spilloverEnd = 3041;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.findName(intArray0, 6);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 8
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("i*PCQb@gXv83@$/^8on", (-1047), 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild((-1089));
      byteQuadsCanonicalizer1._count = 4128;
      byteQuadsCanonicalizer1.addName(" etries, hashBarea o6 ", (-1089));
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName(" etries, hashBarea o6 ", 0, (-1089));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=4129, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1000);
      byteQuadsCanonicalizer1._count = 256;
      byteQuadsCanonicalizer1.addName((String) null, 0, 512);
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName((String) null, 1000, 1000);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Failed rehash(): old count=257, copyCount=1
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(964);
      byteQuadsCanonicalizer1.addName("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", 964, 964, 964);
      byteQuadsCanonicalizer1._hashSize = 2;
      byteQuadsCanonicalizer1.addName("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", 70, 70, 964);
      byteQuadsCanonicalizer1.addName("Eti9Rub", (-58));
      assertEquals(0, byteQuadsCanonicalizer1.spilloverCount());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      int[] intArray0 = new int[22];
      intArray0[3] = 4;
      byteQuadsCanonicalizer0._hashSize = 1;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("", intArray0, 4);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot(680);
      int[] intArray0 = new int[7];
      byteQuadsCanonicalizer0._spilloverEnd = 680;
      intArray0[3] = 680;
      byteQuadsCanonicalizer0._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("[com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer: size=0, hashSize=0, 0/0/0/170 pri/sec/ter/spill (=0), total:170]", 680);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 680
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot((-4274));
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0.addName("U:#HNcDBh", (int[]) null, (-4274));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      ByteQuadsCanonicalizer byteQuadsCanonicalizer1 = byteQuadsCanonicalizer0.makeChild(1000);
      byteQuadsCanonicalizer1._count = 256;
      byteQuadsCanonicalizer1.addName((String) null, 0, 512);
      int[] intArray0 = new int[4];
      intArray0[3] = 1000;
      byteQuadsCanonicalizer1._hashArea = intArray0;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer1.addName((String) null, 1000, 1000);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      ByteQuadsCanonicalizer byteQuadsCanonicalizer0 = ByteQuadsCanonicalizer.createRoot();
      byteQuadsCanonicalizer0._hashSize = 4096;
      // Undeclared exception!
      try { 
        byteQuadsCanonicalizer0._reportTooManyCollisions();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Spill-over slots in symbol table with 0 entries, hash area of 4096 slots is now full (all 512 slots -- suspect a DoS attack based on hash collisions. You can disable the check via `JsonFactory.Feature.FAIL_ON_SYMBOL_HASH_OVERFLOW`
         //
         verifyException("com.fasterxml.jackson.core.sym.ByteQuadsCanonicalizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(375);
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(839877741);
      assertEquals(7, int0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      int int0 = ByteQuadsCanonicalizer._calcTertiaryShift(1609);
      assertEquals(6, int0);
  }
}
