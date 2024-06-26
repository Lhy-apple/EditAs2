/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:10:01 GMT 2023
 */

package org.jsoup.select;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.ListIterator;
import java.util.function.UnaryOperator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeVisitor;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Elements_ESTest extends Elements_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.toArray((Integer[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Elements elements0 = new Elements();
      // Undeclared exception!
      try { 
        elements0.remove((-3893));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Elements elements0 = new Elements();
      LinkedHashSet<Integer> linkedHashSet0 = new LinkedHashSet<Integer>();
      boolean boolean0 = elements0.removeAll(linkedHashSet0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Document.createShell("F1");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.add((-133), (Element) document0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -133, Size: 4
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.remove((Object) elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      ListIterator<Element> listIterator0 = elements0.listIterator(1);
      assertTrue(listIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.containsAll(elements0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Document.createShell("A");
      Elements elements0 = document0.getAllElements();
      elements0.clear();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getAllElements();
      int int0 = elements0.indexOf("\n");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Document.createShell(",r!'[BS?na");
      Elements elements0 = document0.getAllElements();
      int int0 = elements0.lastIndexOf(document0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.not("\n");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("y63");
      Element element0 = new Element(tag0, "\n");
      Elements elements0 = element0.getElementsMatchingText("UFbOD57CQR%");
      elements0.hashCode();
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.toString();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.addAll((-2206), (Collection<? extends Element>) elements0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: -2206, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      UnaryOperator<Element> unaryOperator0 = UnaryOperator.identity();
      elements0.replaceAll(unaryOperator0);
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.subList(2010, 2010);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // toIndex = 2010
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Elements elements0 = new Elements();
      boolean boolean0 = elements0.addAll((Collection<? extends Element>) elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.parents();
      assertNotSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = Document.createShell("F1");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.is("F1");
      assertFalse(boolean0);
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.set(161, (Element) document0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 161, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = Document.createShell("A");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.retainAll(elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.clone();
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.attr("%");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      elements0.attr("\n", "\n");
      // Undeclared exception!
      try { 
        elements0.attr("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasAttr("Ll_BMLmqlu\"$TMU");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      elements0.attr("\n", "\n");
      boolean boolean0 = elements0.hasAttr("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.removeAttr("\n");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = Document.createShell("A");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.addClass("org.jsoup.select.Elements");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document(",r!'[BS?na");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.removeClass(",r!'[BS?na");
      assertEquals(1, elements1.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = Document.createShell("A");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.toggleClass("A");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasClass("\n");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Elements elements0 = new Elements();
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.val("");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = Document.createShell("6\n");
      Elements elements0 = document0.getAllElements();
      elements0.prepend("6\n");
      String string0 = elements0.text();
      assertEquals("6 6 6 6 6 6 6 6 6", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      document0.prepend("'* e!n|c}\"r%6");
      boolean boolean0 = elements0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.html();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>\n\n", string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.tagName("F1");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.html("F1");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.append("F1");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.before("=`2_9&4to|P");
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.before("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.after("");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = Document.createShell("A");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.after("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.wrap("7cb");
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = Document.createShell("E<");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.wrap("E<");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Elements elements0 = new Elements();
      Elements elements1 = elements0.unwrap();
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("F1");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.unwrap();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.empty();
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = Document.createShell("l!");
      Elements elements0 = document0.parents();
      Elements elements1 = elements0.remove();
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Tag tag0 = Tag.valueOf("y63");
      Element element0 = new Element(tag0, "\n");
      Elements elements0 = element0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.remove();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Tag tag0 = Tag.valueOf("y63");
      Element element0 = new Element(tag0, "\n");
      Elements elements0 = element0.getElementsMatchingText("UFbOD57CQR%");
      Elements elements1 = elements0.eq(0);
      assertNotSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Tag tag0 = Tag.valueOf("y63");
      Element element0 = new Element(tag0, "\n");
      Elements elements0 = element0.getAllElements();
      boolean boolean0 = elements0.is("y63");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = Document.createShell("F1");
      String string0 = document0.title();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Tag tag0 = Tag.valueOf("y63");
      Element element0 = new Element(tag0, "\n");
      Elements elements0 = element0.getAllElements();
      Element element1 = elements0.last();
      assertEquals(0, element1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Tag tag0 = Tag.valueOf("y63");
      Element element0 = new Element(tag0, "\n");
      Elements elements0 = element0.getElementsMatchingText("UFbOD57CQR%");
      Element element1 = elements0.last();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getAllElements();
      NodeVisitor nodeVisitor0 = mock(NodeVisitor.class, new ViolatedAssumptionAnswer());
      Elements elements1 = elements0.traverse(nodeVisitor0);
      assertEquals(4, elements1.size());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("leftharhoondoZn");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-222));
      boolean boolean0 = elements0.contains("leftharhoondoZn");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = Document.createShell("\n");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.eq(1);
      boolean boolean0 = elements1.containsAll(elements0);
      assertEquals(1, elements1.size());
      assertFalse(boolean0);
  }
}
