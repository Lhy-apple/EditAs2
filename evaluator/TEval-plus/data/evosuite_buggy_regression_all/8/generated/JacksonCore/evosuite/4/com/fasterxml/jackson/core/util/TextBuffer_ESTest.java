/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:53:59 GMT 2023
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
      String string0 = textBuffer0.toString();
      assertEquals("", string0);
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
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithCopy(charArray0, 1, 2103);
      String string0 = textBuffer0.contentsAsString();
      TextBuffer textBuffer1 = new TextBuffer(bufferRecycler0);
      textBuffer1.resetWithCopy(charArray0, 1500, 0);
      textBuffer1.append(string0, 0, 1500);
      assertEquals(2103, textBuffer0.size());
      assertEquals(1500, textBuffer1.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      textBuffer0.resetWithString((String) null);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      int int0 = textBuffer0.getCurrentSegmentSize();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.getTextOffset());
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      textBuffer0.releaseBuffers();
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 1, 3);
      textBuffer0.ensureNotShared();
      assertEquals(3, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 3, 2);
      assertEquals(3, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      char[] charArray0 = textBuffer0.finishCurrentSegment();
      textBuffer0.resetWithCopy(charArray0, 3, 3);
      assertEquals(3, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithCopy(charArray0, 1, 3164);
      assertEquals(3164, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithString("");
      assertFalse(textBuffer0.hasTextAsCharacters());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      TextBuffer textBuffer0 = new TextBuffer((BufferRecycler) null);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(1000, charArray0.length);
      
      textBuffer0.ensureNotShared();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('(');
      textBuffer0.contentsAsArray();
      assertEquals(1, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.size();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(200, charArray0.length);
      assertNotNull(charArray0);
      
      char[] charArray1 = textBuffer0.contentsAsArray();
      assertNotNull(charArray1);
      
      int int0 = textBuffer0.size();
      assertEquals(0, textBuffer0.getTextOffset());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('(');
      textBuffer0.contentsAsString();
      int int0 = textBuffer0.size();
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      int int0 = textBuffer0.getTextOffset();
      assertEquals(0, textBuffer0.size());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      int int0 = textBuffer0.getTextOffset();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler(729, 729);
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
      assertEquals(0, textBuffer0.getTextOffset());
      assertTrue(boolean0);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.getTextBuffer();
      textBuffer0.hasTextAsCharacters();
      assertEquals(200, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      textBuffer0.contentsAsString();
      boolean boolean0 = textBuffer0.hasTextAsCharacters();
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
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(200, charArray0.length);
      
      textBuffer0.contentsAsArray();
      char[] charArray1 = textBuffer0.getTextBuffer();
      assertEquals(0, charArray1.length);
      assertNotNull(charArray1);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsString();
      textBuffer0.getCurrentSegment();
      assertFalse(textBuffer0.hasTextAsCharacters());
      
      textBuffer0.getTextBuffer();
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      char[] charArray1 = textBuffer0.getTextBuffer();
      assertSame(charArray1, charArray0);
      assertNotNull(charArray1);
      assertEquals(200, charArray1.length);
      assertEquals(0, textBuffer0.size());
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
      char[] charArray0 = textBuffer0.getCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 3, 2);
      textBuffer0.contentsAsString();
      assertEquals(2, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('(');
      textBuffer0.finishCurrentSegment();
      textBuffer0.contentsAsString();
      assertFalse(textBuffer0.hasTextAsCharacters());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.contentsAsArray();
      char[] charArray1 = textBuffer0.contentsAsArray();
      assertSame(charArray1, charArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
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
  public void test34()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append("Or>Vmz0@", 2, 3);
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \">Vm\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler(2166, 2166);
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 2166, (-947));
      // Undeclared exception!
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append("Or>Vmz0@", 2, 3);
      textBuffer0.finishCurrentSegment();
      try { 
        textBuffer0.contentsAsDecimal();
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Value \">Vm\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\" can not be represented as BigDecimal
         //
         verifyException("com.fasterxml.jackson.core.io.NumberInput", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append('(');
      textBuffer0.append('M');
      assertEquals(2, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      // Undeclared exception!
      try { 
        textBuffer0.append((char[]) null, 1, 2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithCopy(charArray0, 1, 2097);
      textBuffer0.append(charArray0, 3, 1);
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithCopy(charArray0, 1, 3153);
      textBuffer0.append("Hj}!}EW/7~t", 3, 1);
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.resetWithEmpty();
      assertEquals(0, textBuffer0.getTextOffset());
      
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(200, charArray0.length);
      assertEquals(0, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      char[] charArray0 = textBuffer0.getCurrentSegment();
      assertEquals(0, textBuffer0.getCurrentSegmentSize());
      assertEquals(200, charArray0.length);
      assertEquals(0, textBuffer0.getTextOffset());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithCopy(charArray0, 1, 2103);
      textBuffer0.getCurrentSegment();
      assertEquals(2103, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.emptyAndGetCurrentSegment();
      textBuffer0.finishCurrentSegment();
      assertEquals(200, textBuffer0.size());
      
      char[] charArray0 = textBuffer0.emptyAndGetCurrentSegment();
      assertEquals(1000, charArray0.length);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      textBuffer0.expandCurrentSegment(262144);
      textBuffer0.finishCurrentSegment();
      assertEquals(262144, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.ensureNotShared();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.finishCurrentSegment();
      textBuffer0.expandCurrentSegment();
      textBuffer0.expandCurrentSegment();
      assertEquals(197512, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithCopy(charArray0, 1, 2103);
      textBuffer0.expandCurrentSegment(45);
      assertEquals(2103, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = textBuffer0.getCurrentSegment();
      textBuffer0.resetWithShared(charArray0, 1, 1);
      // Undeclared exception!
      try { 
        textBuffer0.append("\"%w6fs0!Qv=~Op", 3, 544);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.append("Or>Vmz0@", 2, 3);
      textBuffer0.setCurrentLength(262144);
      textBuffer0.expandCurrentSegment(262144);
      textBuffer0.append('w');
      assertEquals(1, textBuffer0.getCurrentSegmentSize());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.contentsAsString();
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
  public void test51()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      char[] charArray0 = bufferRecycler0.allocCharBuffer(1);
      textBuffer0.resetWithShared(charArray0, 0, 3);
      textBuffer0.contentsAsArray();
      assertEquals(3, textBuffer0.size());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      BufferRecycler bufferRecycler0 = new BufferRecycler();
      TextBuffer textBuffer0 = new TextBuffer(bufferRecycler0);
      textBuffer0.getCurrentSegment();
      char[] charArray0 = textBuffer0.expandCurrentSegment(2);
      textBuffer0.resetWithShared(charArray0, 3, 2);
      textBuffer0.contentsAsArray();
      assertEquals(2, textBuffer0.size());
  }
}