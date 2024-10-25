/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:52:09 GMT 2023
 */

package org.jsoup.select;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.ListIterator;
import java.util.regex.Pattern;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeVisitor;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Elements_ESTest extends Elements_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("lsime");
      boolean boolean0 = elements0.addAll((Collection<? extends Element>) elements0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsMatchingText("dfn");
      String[] stringArray0 = new String[6];
      String[] stringArray1 = elements0.toArray(stringArray0);
      assertSame(stringArray1, stringArray0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("Hwkr9R");
      // Undeclared exception!
      try { 
        elements0.remove((-699));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("Hwkr9R");
      LinkedHashSet<Document> linkedHashSet0 = new LinkedHashSet<Document>();
      boolean boolean0 = elements0.removeAll(linkedHashSet0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getElementsContainingText("7Nn:~R\"");
      boolean boolean0 = elements0.equals("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Elements elements0 = new Elements();
      ListIterator<Element> listIterator0 = elements0.listIterator(0);
      assertFalse(listIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.remove((Object) document0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("Hwkr9R");
      boolean boolean0 = elements0.is("\";w!8AWh~gJR");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("Hwkr9R");
      elements0.clear();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Element[] elementArray0 = new Element[1];
      Elements elements0 = new Elements(elementArray0);
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.parents();
      int int0 = elements0.indexOf("[1F=w*VFK:=NkCC");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsByAttribute("org.jsoup.select.Elements");
      int int0 = elements0.lastIndexOf("pdRTmIh*");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("lsime");
      Elements elements1 = elements0.not("pdRTmIh*");
      assertEquals(0, elements1.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("#a_z");
      Pattern pattern0 = Pattern.compile("v- >'G0-]l");
      Elements elements0 = document0.getElementsByAttributeValueMatching("", pattern0);
      elements0.hashCode();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsByAttribute("org.jsoup.select.Elements");
      String string0 = elements0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getElementsContainingText("7Nn:~R\"");
      // Undeclared exception!
      try { 
        elements0.addAll(1058, (Collection<? extends Element>) elements0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1058, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = Document.createShell(";FtILAov");
      Elements elements0 = document0.getElementsContainingOwnText(";FtILAov");
      ListIterator<Element> listIterator0 = elements0.listIterator();
      assertFalse(listIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByAttributeValueEnding("\n", "\n");
      // Undeclared exception!
      try { 
        elements0.subList(1, 1);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // toIndex = 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Elements elements0 = new Elements();
      Document document0 = Document.createShell("svg");
      // Undeclared exception!
      try { 
        elements0.set(1, (Element) document0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Elements elements0 = new Elements();
      LinkedHashSet<Object> linkedHashSet0 = new LinkedHashSet<Object>();
      boolean boolean0 = elements0.retainAll(linkedHashSet0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = Document.createShell("(71$");
      Elements elements0 = document0.getElementsByAttributeValueContaining("nHGTSL>", "(71$");
      elements0.add(0, (Element) document0);
      Elements elements1 = elements0.clone();
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.attr("pdRTmIh*");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.attr("[1F=w*VFK:=NkCC", "[1F=w*VFK:=NkCC");
      String string0 = elements1.attr("[1F=w*VFK:=NkCC");
      assertEquals("[1F=w*VFK:=NkCC", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasAttr(">5T}%)<a");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getAllElements();
      elements0.attr("+MH<vxJP)#-k", "+MH<vxJP)#-k");
      boolean boolean0 = elements0.hasAttr("+MH<vxJP)#-k");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("\n");
      Elements elements0 = document0.getElementsByAttributeValueEnding("\n", "\n");
      Elements elements1 = elements0.removeAttr((String) null);
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = Document.createShell("(71$");
      Elements elements0 = document0.getElementsByAttributeValueContaining("nHGTSL>", "(71$");
      elements0.add(0, (Element) document0);
      // Undeclared exception!
      try { 
        elements0.removeAttr("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.addClass("[1F=w*VFK:=NkCC");
      assertEquals(1, elements1.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.removeClass("pdRTmIh*");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = Document.createShell("(71$");
      Elements elements0 = document0.getElementsByAttributeValueContaining("FHGTSL>", "(71$");
      Elements elements1 = elements0.toggleClass("FHGTSL>");
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getElementsMatchingText("");
      // Undeclared exception!
      try { 
        elements0.toggleClass((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasClass("+MH<vxJP)#-k");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsByAttribute("org.jsoup.select.Elements");
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.val("[1F=w*VFK:=NkCC");
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("SqOBT1DXzuhMfc}");
      document0.normalise();
      Element element0 = document0.text("kbd");
      Elements elements0 = element0.getElementsContainingText("kbd");
      String string0 = elements0.text();
      assertEquals("kbd kbd kbd", string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = Document.createShell("");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsContainingOwnText("");
      elements0.html("YgWPy@}DZlA");
      boolean boolean0 = elements0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsContainingOwnText("");
      String string0 = elements0.html();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>\n\n", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      String string0 = elements0.outerHtml();
      assertEquals("<html>\n <head></head>\n <body></body>\n</html>\n<html>\n <head></head>\n <body></body>\n</html>\n<head></head>\n<body></body>", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("1c0F=w*VF6:=kCC");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.tagName("1c0F=w*VF6:=kCC");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmlh]");
      Elements elements0 = document0.getElementsContainingOwnText("");
      Elements elements1 = elements0.prepend("\n");
      assertEquals(4, elements1.size());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.append("");
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsMatchingText("dfn");
      Elements elements1 = elements0.before("dfn");
      assertEquals(0, elements1.size());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsContainingOwnText("");
      // Undeclared exception!
      try { 
        elements0.before("pdRTmIh*");
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
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsByAttribute("org.jsoup.select.Elements");
      Elements elements1 = elements0.after("org.jsoup.select.Elements");
      assertTrue(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("sfr");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.after("sfr");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = Document.createShell("looparrowleft");
      Elements elements0 = document0.getElementsMatchingText("looparrowleft");
      Elements elements1 = elements0.wrap("looparrowleft");
      assertEquals(0, elements1.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("1");
      Elements elements0 = document0.getAllElements();
      // Undeclared exception!
      try { 
        elements0.wrap("(!|dDhz7<T\"vvRi7");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getElementsMatchingText("dfn");
      Elements elements1 = elements0.unwrap();
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
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
  public void test52()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.empty();
      assertSame(elements0, elements1);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = Document.createShell("(71$");
      Elements elements0 = document0.getElementsByAttributeValueContaining("nHGTSL>", "(71$");
      Elements elements1 = elements0.remove();
      assertSame(elements1, elements0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.getAllElements();
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
  public void test55()  throws Throwable  {
      Document document0 = Document.createShell("(71$");
      Elements elements0 = document0.getElementsByAttributeValueContaining("nHGTSL>", "(71$");
      Elements elements1 = elements0.eq(0);
      assertTrue(elements1.equals((Object)elements0));
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsByAttribute("org.jsoup.select.Elements");
      // Undeclared exception!
      try { 
        elements0.eq((-3708));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.is(",9");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getAllElements();
      Elements elements1 = elements0.parents();
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.nodes.Entities");
      String string0 = document0.title();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getAllElements();
      Element element0 = elements0.last();
      assertEquals("", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsByAttribute("org.jsoup.select.Elements");
      Element element0 = elements0.last();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Document document0 = Document.createShell("[1F=w*VFK:=NkCC");
      Elements elements0 = document0.children();
      NodeVisitor nodeVisitor0 = mock(NodeVisitor.class, new ViolatedAssumptionAnswer());
      Elements elements1 = elements0.traverse(nodeVisitor0);
      assertFalse(elements1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("Hwkr9R");
      boolean boolean0 = elements0.contains("L50y%lha9<=L");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = new Document("");
      Elements elements0 = document0.getAllElements();
      boolean boolean0 = elements0.contains(document0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("pdRTmIh*");
      Elements elements1 = document0.getAllElements();
      boolean boolean0 = elements0.containsAll(elements1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Document document0 = Document.createShell("pdRTmIh*");
      Elements elements0 = document0.getElementsMatchingText("Hwkr9R");
      boolean boolean0 = elements0.containsAll(elements0);
      assertTrue(boolean0);
  }
}
