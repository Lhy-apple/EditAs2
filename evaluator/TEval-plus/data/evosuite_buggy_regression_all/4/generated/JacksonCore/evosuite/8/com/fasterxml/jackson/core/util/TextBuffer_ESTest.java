/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:36:22 GMT 2023
 */

package com.fasterxml.jackson.core.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.core.util.BufferRecycler;
import com.fasterxml.jackson.core.util.TextBuffer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TextBuffer_ESTest extends TextBuffer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.toString();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      try { 
        textBuffer0.contentsAsDouble();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 0, 2);
      assertEquals(2, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.getCurrentSegmentSize();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.getTextOffset());
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('L');
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      textBuffer0.resetWithEmpty();
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 1, 2);
      textBuffer0.getCurrentSegment();
      assertEquals(2, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithCopy(charArray0, 3, 1);
      assertEquals(1, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithCopy(charArray0, 1, 3);
      assertEquals(3, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      // Undeclared exception!
      try { 
        textBuffer0.resetWithCopy((char[]) null, 3, 1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithString("5");
      assertEquals(1, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithString("7&dj9<)-fxj32[D");
      assertFalse(textBuffer0.hasTextAsCharacters());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      textBuffer0.ensureNotShared();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(200, charArray0.length);
      
      char[] charArray1 = textBuffer0.contentsAsArray();
      assertNotNull(charArray1);
      
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertEquals(0, textBuffer0.getTextOffset());
      assertTrue(boolean0);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler(55, 55);
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.size();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append(':');
      textBuffer0.contentsAsString();
      textBuffer0.getTextBuffer();
      int int0 = textBuffer0.size();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.contentsAsString();
      int int0 = textBuffer0.size();
      assertFalse(textBuffer0.hasTextAsCharacters());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      int int0 = textBuffer0.getTextOffset();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.getTextOffset();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append(':');
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append(':');
      textBuffer0.contentsAsString();
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getTextBuffer();
      assertNull(charArray0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(200, charArray0.length);
      
      textBuffer0.contentsAsArray();
      char[] charArray1 = textBuffer0.getTextBuffer();
      assertEquals(0, charArray1.length);
      assertEquals(0, textBuffer0.size());
      assertNotNull(charArray1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = textBuffer0.getTextBuffer();
      assertEquals(200, charArray0.length);
      assertNotNull(charArray0);
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.getTextBuffer();
      assertEquals(200, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsString();
      String string0 = textBuffer0.contentsAsString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsArray();
      String string0 = textBuffer0.contentsAsString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 0, 2);
      textBuffer0.toString();
      assertEquals(2, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.contentsAsArray();
      char[] charArray1 = textBuffer0.contentsAsArray();
      assertSame(charArray1, charArray0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsArray();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append("d6%>c%B4>g", 0, 7);
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"d6%>c%B\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 420, 660);
      // Undeclared exception!
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append("d6%>c%B4>g", 0, 7);
      textBuffer0.finishCurrentSegment();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \"d6%>c%B\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.ensureNotShared();
      assertEquals(1, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.setCurrentLength(102514);
      textBuffer0.append('R');
      textBuffer0.setCurrentAndReturn(587);
      assertEquals(587, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      char[] charArray0 = textBuffer0.expandCurrentSegment(4161);
      TextBuffer textBuffer1 = new TextBuffer(bufferRecycler0);
      textBuffer1.append(charArray0, 4161, 0);
      textBuffer1.append(charArray0, 1, 1933);
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertEquals(1933, textBuffer1.size());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.setCurrentLength(4161);
      char[] charArray0 = textBuffer0.expandCurrentSegment(4161);
      textBuffer0.append(charArray0, 1, 1933);
      assertEquals(1933, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.append("", 0, 0);
      assertEquals(1, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      String string0 = textBuffer0.contentsAsString();
      // Undeclared exception!
      try { 
        textBuffer0.append(string0, 1, 102514);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.setCurrentLength(4161);
      textBuffer0.append("|YmVM1}]qZ=m[V+' .", 0, 3);
      assertEquals(3, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.expandCurrentSegment(16123);
      textBuffer0.setCurrentLength(102514);
      TextBuffer textBuffer1 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      String string0 = textBuffer0.contentsAsString();
      textBuffer1.getCurrentSegment();
      // Undeclared exception!
      try { 
        textBuffer1.append(string0, 1, 102514);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(0, textBuffer0.getTextOffset());
      
      char[] charArray1 = textBuffer0.getCurrentSegment();
      assertSame(charArray1, charArray0);
      assertEquals(200, charArray1.length);
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithEmpty();
      assertEquals(0, textBuffer0.getTextOffset());
      
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(200, charArray0.length);
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('R');
      textBuffer0.setCurrentLength(3673);
      textBuffer0.getCurrentSegment();
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(1000, charArray0.length);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      String string0 = textBuffer0.setCurrentAndReturn(0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      // Undeclared exception!
      try { 
        textBuffer0.setCurrentAndReturn(3);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(1200, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.expandCurrentSegment(262144);
      textBuffer0.finishCurrentSegment();
      assertEquals(262144, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(200, charArray0.length);
      
      char[] charArray1 = textBuffer0.expandCurrentSegment(262144);
      assertEquals(262144, charArray1.length);
      
      char[] charArray2 = textBuffer0.expandCurrentSegment();
      assertEquals(0, textBuffer0.size());
      assertEquals(327680, charArray2.length);
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = textBuffer0.expandCurrentSegment(0);
      assertEquals(0, textBuffer0.getTextOffset());
      assertEquals(200, charArray0.length);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 1, 991);
      // Undeclared exception!
      try { 
        textBuffer0.append(charArray0, 1000, 3);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('!');
      textBuffer0.expandCurrentSegment(16202);
      textBuffer0.setCurrentLength(102514);
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.append('!');
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.setCurrentLength(262144);
      textBuffer0.append('0');
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithShared((char[]) null, 0, 1);
      // Undeclared exception!
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Arrays", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithShared((char[]) null, 3, 2);
      // Undeclared exception!
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Arrays", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('B');
      textBuffer0.contentsAsArray();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
  }
}
