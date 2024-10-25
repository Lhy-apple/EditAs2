/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:54:16 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.jdom;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.compiler.NodeNameTest;
import org.apache.commons.jxpath.ri.compiler.NodeTest;
import org.apache.commons.jxpath.ri.compiler.NodeTypeTest;
import org.apache.commons.jxpath.ri.compiler.ProcessingInstructionTest;
import org.apache.commons.jxpath.ri.model.NodeIterator;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jdom.Attribute;
import org.jdom.CDATA;
import org.jdom.Comment;
import org.jdom.Content;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.ProcessingInstruction;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class JDOMNodePointer_ESTest extends JDOMNodePointer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isCollection();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodePointer nodePointer0 = jDOMNodePointer0.namespacePointer("Tyo");
      assertEquals(Integer.MIN_VALUE, nodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = Locale.CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      NodeIterator nodeIterator0 = jDOMNodePointer0.childIterator(nodeTypeTest0, true, (NodePointer) null);
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      int int0 = jDOMNodePointer0.getLength();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      CDATA cDATA0 = new CDATA("[+0[");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest((String) null);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeIterator nodeIterator0 = jDOMNodePointer0.namespaceIterator();
      assertEquals(0, nodeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, locale0, "_Tyo");
      int int0 = jDOMNodePointer0.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer1);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "_Tyo");
      Locale locale0 = jXPathContext0.getLocale();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
      assertEquals("_Tyo", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Element element0 = new Element("MOf");
      CDATA cDATA0 = new CDATA("MOf");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, (Locale) null);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, locale0);
      jDOMNodePointer0.hashCode();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Locale locale0 = Locale.ITALY;
      Integer integer0 = new Integer(1908);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(integer0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild((JXPathContext) null, qName0, 1908, (Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document();
      Locale locale0 = new Locale("r[ZilRgw");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0, "r[ZilRgw");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.getNamespaceURI("r[ZilRgw");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root element not set
         //
         verifyException("org.jdom.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("Tyo");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Element element0 = new Element("MOf", "MOf", "MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getNamespaceURI("<<unknown namespace>>");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = new Object();
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(object0, locale0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.compareChildNodePointers(jDOMNodePointer0, jDOMNodePointer1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // JXPath internal error: compareChildNodes called for java.lang.Object@347b47fe
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document();
      Locale locale0 = Locale.TRADITIONAL_CHINESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(document0, locale0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.isLeaf();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Root element not set
         //
         verifyException("org.jdom.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Element element0 = new Element("MOf", "MOf", "MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Element element0 = new Element("MOf", "MOf", "MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue("MOf");
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLeaf();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Element element0 = new Element("fOf", "fOf", "fOf");
      Locale locale0 = Locale.ROOT;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("fOf:fOf", nodeNameTest0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("chil", "chil");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      assertNull(qName0.getPrefix());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("chil", "chil");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNotNull(object0);
      assertEquals("chil", object0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("", object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Locale locale0 = Locale.JAPAN;
      Comment comment0 = new Comment("\"Wo>=\"");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("\"Wo>=\"", object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CDATA cDATA0 = new CDATA("~Y>PGm/1S&");
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0, "~Y>PGm/1S&");
      Object object0 = jDOMNodePointer0.getValue();
      assertEquals("~Y>PGm/1S&", object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      Object object0 = jDOMNodePointer0.getValue();
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      CDATA cDATA0 = new CDATA("[+0[");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, locale0);
      LinkedList<String> linkedList0 = new LinkedList<String>();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue(linkedList0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      CDATA cDATA0 = new CDATA("Of");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, (Locale) null);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.setValue((Object) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      CDATA cDATA0 = new CDATA("http://www.w3.org/XML/1998/namespace");
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, cDATA0);
      jDOMNodePointer1.setValue(cDATA0);
      assertFalse(jDOMNodePointer1.isAttribute());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue(element0);
      assertFalse(element0.isRootElement());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      CDATA cDATA0 = new CDATA((String) null);
      jDOMNodePointer0.setValue(cDATA0);
      assertEquals(Integer.MIN_VALUE, jDOMNodePointer0.getIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "_Tyo");
      Locale locale0 = jXPathContext0.getLocale();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      Comment comment0 = new Comment("&quot;");
      jDOMNodePointer0.setValue(comment0);
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      jDOMNodePointer0.setValue((Object) null);
      assertEquals(Integer.MIN_VALUE, NodePointer.WHOLE_COLLECTION);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Element element0 = new Element("MOf");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, (Locale) null);
      jDOMNodePointer0.setValue("");
      assertEquals(1, jDOMNodePointer0.getLength());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      QName qName0 = new QName("~Y>PGm/1S&", "~Y>PGm/1S&");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "~Y>PGm/1S&");
      boolean boolean0 = JDOMNodePointer.testNode((NodePointer) null, (Object) qName0, (NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Element element0 = new Element("MOf", "MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("http://www.w3.org/2000/xmlns/", "<<unknown namespace>>");
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(Integer.MIN_VALUE);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Element element0 = new Element("Mf", "Mf");
      NodeTypeTest nodeTypeTest0 = new NodeTypeTest(1);
      Locale locale0 = Locale.KOREA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeTypeTest0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("ch3vil", "ch3vil");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      ProcessingInstructionTest processingInstructionTest0 = new ProcessingInstructionTest("<<unknown namespace>>");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) processingInstructionTest0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.ITALIAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0);
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("MOf", nodeNameTest0.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      NodeNameTest nodeNameTest0 = new NodeNameTest(qName0, "http://www.w3.org/2000/xmlns/");
      boolean boolean0 = jDOMNodePointer0.testNode((NodeTest) nodeNameTest0);
      assertEquals("MOf", nodeNameTest0.toString());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Locale locale0 = Locale.CANADA;
      String string0 = JDOMNodePointer.getPrefix(locale0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Element element0 = new Element("_TSyo", "_TSyo", "_TSyo");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertEquals("_TSyo", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Element element0 = new Element("_Tyo", "_Tyo");
      String string0 = JDOMNodePointer.getPrefix(element0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Attribute attribute0 = new Attribute("K", "K");
      String string0 = JDOMNodePointer.getLocalName(attribute0);
      assertEquals("K", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      String string0 = JDOMNodePointer.getLocalName("@(#) $RCSfile: AttributeList.java,v $ $Revision: 1.23 $ $Date: 2004/02/28 03:30:27 $ $Name: jdom_1_0 $");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("http://www.w3.org/XML/1998/namespace");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) "_Tyo");
      Locale locale0 = jXPathContext0.getLocale();
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.getLanguage();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("chil", "chil");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("chil");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Comment comment0 = new Comment("d5]{r{=|*+n[a");
      Locale locale0 = Locale.JAPANESE;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(comment0, locale0);
      boolean boolean0 = jDOMNodePointer0.isLanguage("http://www.w3.org/XML/1998/namespace");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createChild(jXPathContext0, qName0, Integer.MIN_VALUE);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Factory is not set on the JXPathContext - cannot create path: 
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Locale locale0 = Locale.UK;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JXPathContext jXPathContext0 = JXPathContext.newContext((Object) jDOMNodePointer0);
      QName qName0 = jDOMNodePointer0.getName();
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute(jXPathContext0, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot create an attribute for path /@null, operation is not allowed for this type of node
         //
         verifyException("org.apache.commons.jxpath.ri.model.NodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Element element0 = new Element("MOf", "MOf", "MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = jDOMNodePointer0.getName();
      jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
      assertEquals("MOf:MOf", qName0.toString());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      QName qName0 = new QName("MOf", "MOf");
      // Undeclared exception!
      try { 
        jDOMNodePointer0.createAttribute((JXPathContext) null, qName0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Unknown namespace prefix: MOf
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Element element0 = new Element("MOf");
      CDATA cDATA0 = new CDATA("MOf");
      element0.addContent((Content) cDATA0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, (Locale) null);
      jDOMNodePointer0.remove();
      assertFalse(jDOMNodePointer0.isCollection());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Element element0 = new Element("fOf", "fOf", "fOf");
      Locale locale0 = Locale.ITALY;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      // Undeclared exception!
      try { 
        jDOMNodePointer0.remove();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot remove root JDOM node
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Element element0 = new Element("_Tyo");
      Locale locale0 = Locale.JAPAN;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0, "\"ik-ar!dH_jP'Dnm.");
      String string0 = jDOMNodePointer0.toString();
      assertEquals("id('&quot;ik-ar!dH_jP&apos;Dnm.')", string0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(element0, locale0);
      String string0 = jDOMNodePointer0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Element element0 = new Element("MOf");
      Locale locale0 = Locale.CANADA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, element0);
      // Undeclared exception!
      try { 
        jDOMNodePointer1.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.jxpath.ri.model.jdom.JDOMNodePointer", e);
      }
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Locale locale0 = Locale.JAPANESE;
      ProcessingInstruction processingInstruction0 = new ProcessingInstruction("chil", "chil");
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(processingInstruction0, locale0);
      String string0 = jDOMNodePointer0.toString();
      assertEquals("/processing-instruction('chil')[1]", string0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Element element0 = new Element("Of");
      CDATA cDATA0 = new CDATA("Of");
      element0.setContent((Content) cDATA0);
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(cDATA0, (Locale) null);
      String string0 = jDOMNodePointer0.toString();
      assertEquals("/text()[1]", string0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer((NodePointer) jDOMNodePointer0, (Object) locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Locale locale0 = Locale.CHINA;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer((Object) null, (Locale) null);
      boolean boolean0 = jDOMNodePointer0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Locale locale0 = Locale.FRENCH;
      JDOMNodePointer jDOMNodePointer0 = new JDOMNodePointer(locale0, locale0);
      JDOMNodePointer jDOMNodePointer1 = new JDOMNodePointer(jDOMNodePointer0, locale0, "http://www.w3.org/XML/1998/namespace");
      boolean boolean0 = jDOMNodePointer0.equals(jDOMNodePointer1);
      assertFalse(boolean0);
  }
}
